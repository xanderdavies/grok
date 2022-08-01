from collections import deque
import wandb
import argparse
from tqdm import tqdm
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from dataset import ArithmeticDataset, ArithmeticIterator
from adamw import AdamW
from open_ai_transformer import Transformer
from simplicity import Simplicity
from margin import get_confidence, get_margin_lists, investigate_param_scaling

"""
Argument Parsing
"""
parser = argparse.ArgumentParser(description="Replication of grokking behavior observed in Power et al.'s 'Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets")

# model hyperparameters
parser.add_argument("--num-layers", default=2, type=int, help="Number of layers in the transformer")
parser.add_argument("--num-heads", default=1, type=int, help="Number of attention heads per layer")
parser.add_argument("--d-model", default=128, type=int, help="Dimension of the model")

# training hyperparameters
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--weight-decay", default=1e-5, type=float, help="Weight decay")
parser.add_argument("--beta1", default=0.9, type=float, help="AdamW beta1")
parser.add_argument("--beta2", default=0.98, type=float, help="AdamW beta2")
parser.add_argument("--use-sgd", default=False, action="store_true")
parser.add_argument("--full-batch", default=False, action="store_true")
parser.add_argument("--momentum", type=float, default=0)

# data hyperparameters 
parser.add_argument("--vocab-len", default=2000, type=int, help="Transformer vocab length")
parser.add_argument("--train-split", default=50, type=int, help="Train split")
parser.add_argument("--embedding-noise", default=0, type=float, help="Add noise to the embedding (value e.g., 0.1)")

# run hyperparameters
parser.add_argument("--optimization-budget", default=3e10, type=int, help="Number of training steps to run")
parser.add_argument("--wandb-project", default="grokking", type=str, help="Wandb project name")
parser.add_argument("--no-logging", action="store_true", help="Disable logging to wandb")
parser.add_argument("--device", default=None, type=str, help="Device used for training.")
parser.add_argument("--resume-run-id", default=None, type=str, help="WandB run to resume.")
parser.add_argument("--load-path", default=None, type=str, help="Load this model.")

args = parser.parse_args()
OPTIMIZATION_BUDGET = args.optimization_budget
LOG = not args.no_logging
DEVICE = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

"""
Dataset
"""
train_dataset, val_dataset = ArithmeticDataset.splits(
    train_pct=args.train_split, 
    operator="/",
    operand_length=None,
)
train_dataloader = ArithmeticIterator(
    train_dataset,
    DEVICE,
    batchsize_hint= (0 if not args.full_batch else -1), # 0 -> default (512), -1 -> full batch
)
val_dataloader = ArithmeticIterator(
    val_dataset,
    DEVICE,
    batchsize_hint=-1,
)

full_train_X, full_train_y = None, None
for data in train_dataloader:
    X, y = data["text"], data["target"]
    if full_train_X is None:
        full_train_X = X
        full_train_y = y
    else:
        full_train_X = torch.cat((full_train_X, X), dim=0)
        full_train_y = torch.cat((full_train_y, y), dim=0)

"""
Model
"""
# decoder-only transfrormer with causal attention masking; 2 layers, width 128, 4 attention heads
model = Transformer(
    n_layers=args.num_layers,
    n_heads=args.num_heads,
    d_model=args.d_model,
    non_linearity="relu",
    vocab_len=args.vocab_len,

).float().to(DEVICE)
if args.load_path is not None:
    model.load_state_dict(torch.load(args.load_path))

"""
WandB Logging
"""
tags = [f"d_model={args.d_model}", f"num_layers={args.num_layers}", f"num_heads={args.num_heads}"]
name = f"split_{args.train_split}-decay_{args.weight_decay}-dim_{args.d_model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
if args.use_sgd:
    name = f"sgd_lr_{args.lr}_mom_{args.momentum}-" + name
    tags = [f"sgd_lr_{args.lr}_mom_{args.momentum}"] + tags
if args.full_batch:
    name = "full_batch-" + name
    tags = ["full_batch"] + tags
if LOG:
    if args.resume_run_id is None:
        wandb.init(project=args.wandb_project, settings=wandb.Settings(start_method="thread"), tags=tags, name=name, config=args)
    else:
        wandb.init(id=args.resume_run_id, resume="must", project=args.wandb_project, settings=wandb.Settings(start_method="thread"), tags=tags, name=name, config=args)

# log number of parameters 
num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
if LOG:
    wandb.log({"Number of Parameters": num_params})
    wandb.watch(model)
print(f"Model has {num_params} trainable parameters.")

# mkae weights directory if needed
try:
    os.makedirs("weights")
except:
    pass

"""
Optimizer
"""
if not args.use_sgd:
    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        betas=(args.beta1, args.beta2)
    )
else:
    optimizer = SGD(
        model.parameters(),
        lr = args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

"""
Criterion
"""
class SpecialCEL(nn.Module):
    @staticmethod
    # from neel nanda's code
    def cross_entropy_high_precision(logits, labels):
        # Shapes: batch x vocab, batch
        # Cast logits to float64 because log_softmax has a float32 underflow on overly 
        # confident data and can only return multiples of 1.2e-7 (the smallest float x
        # such that 1+x is different from 1 in float32). This leads to loss spikes 
        # and dodgy gradients
        logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
        prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
        loss = -torch.mean(prediction_logprobs)
        return loss
    def forward(self, input, target):
        return self.cross_entropy_high_precision(input[:, -2, :], target[:, -2]) 
        
criterion = SpecialCEL()

"""
Train
"""
steps_per_epoch = len(train_dataloader)
loss_peak = -1
for epoch in tqdm(range(int(OPTIMIZATION_BUDGET / steps_per_epoch))):
    # train
    model.train()
    margin, margin_norms = [], []
    for i, batch in enumerate(train_dataloader):
        X, y = batch['text'], batch['target']
        X, y = X.to(DEVICE), y.to(DEVICE)
        y_hat = model(X, embedding_noise=args.embedding_noise)
        loss = criterion(y_hat, y)
        train_acc = (y_hat[:, -2, :].argmax(dim=1) == y[:, -2]).float().mean().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            batch_margins, batch_margin_norms = get_margin_lists(y_hat[:, -2, :], y[:, -2])
            margin += batch_margins; margin_norms += batch_margin_norms
            if LOG:
                *_, last_weights = model.parameters()
                conf, conf_norm = get_confidence(y_hat[:, -2, :], y[:, -2])
                log_dict = {
                    "Loss/train": loss.item(), 
                    "Accuracy/train": train_acc,
                    "Margins/Average Margin": torch.mean(torch.tensor(margin)).item(),
                    "Margins/Average Margin (normalized)": torch.mean(torch.tensor(margin_norms)).item(),
                    "Margins/Minimum Margin": torch.min(torch.tensor(margin)).item(),
                    "Margins/Minimum Margin (normalized)": torch.min(torch.tensor(margin_norms)).item(),
                    "Margins/Maximum Margin": torch.max(torch.tensor(margin)).item(),
                    "Margins/Maximum Margin (normalized)": torch.max(torch.tensor(margin_norms)).item(),
                    "Simplicity/norm of last weights": torch.linalg.norm(last_weights).item(),
                    "Confidence/average confidence": torch.mean(conf).item(),
                    "Confidence/average confidence (normalized)": torch.mean(conf_norm).item(),
                    "Confidence/minimum confidence": torch.min(conf).item(),
                    "Confidence/minimum confidence (normalized)": torch.min(conf_norm).item(),
                    "epoch": epoch,
                }
                wandb.log(log_dict)
            else:
                print(f"Epoch {epoch}: train loss {loss.item()}, train accuracy {train_acc}")

    # eval
    model.eval()
    with torch.no_grad():
        loss, accuracy = 0, 0
        for batch in val_dataloader: # only one batch
            X, y = batch['text'], batch['target']
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_hat = model(X)
            loss += criterion(y_hat, y).item()
            accuracy += (y_hat[:, -2, :].argmax(dim=1) == y[:, -2]).float().mean().item()
        if LOG:
            log_dict = {
                "Loss/val": loss / len(val_dataloader),
                "Accuracy/val": accuracy / len(val_dataloader),
                "epoch": epoch,
            }
            log_dict.update(Simplicity(model, criterion, (full_train_X, full_train_y), DEVICE).random_noise_metric())
            wandb.log(log_dict)
        else:
            print(f"Epoch {epoch}: test loss {loss / len(val_dataloader)}, test accuracy {accuracy / len(val_dataloader)}")
    
    # These take a long time, so are off for now.
    # if LOG and epoch % 10 == 0:
    #     simplicity = Simplicity(model, criterion, (full_train_X, full_train_y), DEVICE)
    #     simp_dict = simplicity.hessian_based_metrics()
    #     simp_dict.update(simplicity.random_noise_metric())
    #     simp_dict.update({"epoch": epoch})
    #     wandb.log(simp_dict)
    #     # investigate_param_scaling(full_train_X, full_train_y, model)

    # save model
    torch.save(model.state_dict(), f"weights/{name}-LATEST-model.pt")