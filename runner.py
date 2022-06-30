import wandb
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from dataset import ArithmeticDataset, ArithmeticIterator
from adamw import AdamW
from open_ai_transformer import Transformer


parser = argparse.ArgumentParser(description="Replication of grokking behavior observed in Power et al.'s 'Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets")
parser.add_argument("--optimization-budget", default=3e10, type=int, help="Number of training steps to run")
parser.add_argument("--wandb-project", default="grokking", type=str, help="Wandb project name")
parser.add_argument("--no-logging", action="store_true", help="Disable logging to wandb")
parser.add_argument("--num-layers", default=2, type=int, help="Number of layers in the transformer")
parser.add_argument("--num-heads", default=1, type=int, help="Number of attention heads per layer")
parser.add_argument("--d-model", default=128, type=int, help="Dimension of the model")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--weight-decay", default=1e-5, type=float, help="Weight decay")
parser.add_argument("--beta1", default=0.9, type=float, help="AdamW beta1")
parser.add_argument("--beta2", default=0.98, type=float, help="AdamW beta2")
parser.add_argument("--vocab-len", default=2000, type=int, help="Transformer vocab length")
parser.add_argument("--device", default=None, type=str, help="Device used for training.")
parser.add_argument("--label-smoothing", default=0, type=float, help="Label smoothing passed to CrossEntropyLoss.")
parser.add_argument("--only-step-when-imperfect", default=False, action="store_true", help="Only take optimizer steps when the train accuracy is imperfect.")
parser.add_argument("--use-sgd", default=False, action="store_true")
parser.add_argument("--full-batch", default=False, action="store_true")
parser.add_argument("--momentum", type=float, default=0)
args = parser.parse_args()

OPTIMIZATION_BUDGET = args.optimization_budget
LOG = not args.no_logging
DEVICE = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

if LOG:
    tags = [f"d_model={args.d_model}", f"num_layers={args.num_layers}", f"num_heads={args.num_heads}", f"smooth={args.label_smoothing}"]
    name = f"dim_{args.d_model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if args.label_smoothing != 0:
        name = f"smooth_{args.label_smoothing}-" + name
        tags = [f"smooth_{args.label_smoothing}"] + tags
    if args.only_step_when_imperfect:
        name = "only_imperfect-" + name
        tags = ["only_imperfect"] + tags
    if args.use_sgd:
        name = f"sgd_lr_{args.lr}_mom_{args.momentum}-" + name
        tags = [f"sgd_lr_{args.lr}_mom_{args.momentum}"] + tags
    if args.full_batch:
        name = "full_batch-" + name
        tags = ["full_batch"] + tags
    wandb.init(project=args.wandb_project, settings=wandb.Settings(start_method="thread"), tags=tags, name=name, config=args)

# get dataloaders
train_dataset, val_dataset = ArithmeticDataset.splits(
        train_pct=50, # careful not to use .5
        operator="/",
        operand_length=None,
    )
train_dataloader = ArithmeticIterator(
        train_dataset,
        DEVICE,
        batchsize_hint= (0 if not args.full_batch else -1), # default (512)
    )
val_dataloader = ArithmeticIterator(
    val_dataset,
    DEVICE,
    batchsize_hint=-1, # just one batch
)

# get model: decoder-only transfrormer with causal attention masking; 2 layers, width 128, 4 attention heads
model = Transformer(
    n_layers=args.num_layers,
    n_heads=args.num_heads,
    d_model=args.d_model,
    non_linearity="relu",
    vocab_len=args.vocab_len,
).float().to(DEVICE)
*_, last_weights = model.parameters()
num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
if LOG:
    wandb.log({"Number of Parameters": num_params})
    wandb.watch(model)
print(f"Model has {num_params} trainable parameters.")

# get optimizer
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

# get criterion
criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

# train model
steps_per_epoch = len(train_dataloader)
step_perf_train_acc, steps_to_98_val, steps_to_100_val = -1, -1, -1
for epoch in tqdm(range(int(OPTIMIZATION_BUDGET / steps_per_epoch))):
    # train
    model.train()

    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        X, y = batch['text'], batch['target']
        X, y = X.to(DEVICE), y.to(DEVICE)
        y_hat, _, _ = model(X)
        loss = criterion(y_hat[:, -2, :], y[:, -2])
        train_acc = (y_hat[:, -2, :].argmax(dim=1) == y[:, -2]).float().mean().item()
        if train_acc == 1 and step_perf_train_acc == -1:
            step_perf_train_acc = epoch*len(train_dataloader) + i
        if train_acc != 1 or not args.only_step_when_imperfect:
            loss.backward()
            optimizer.step()
        else:
            print("not stepping!")
        with torch.no_grad():
            if LOG:
                soft_out =  F.softmax(y_hat[:, -2, :], dim=1).max(dim=1)[0].cpu().numpy()
                log_dict = {
                    "Loss/train": loss.item(), 
                    "epoch": epoch,
                    "Accuracy/train": train_acc,
                    "train max(softmax(out))": soft_out,
                    "average train max(softmax(out))": soft_out.mean().item(),
                    "min train max(softmax(out))": soft_out.min().item(),
                    "norm of last weights": torch.linalg.norm(last_weights).item(),
                }
                if i == 0:
                    cum_l2_norm, cum_l1_norm, cum_inf_norm = 0, 0, 0
                    for p in [param for param in model.parameters() if param.requires_grad]:
                        cum_l2_norm += torch.linalg.norm(p)
                        cum_l1_norm += torch.linalg.norm(p, ord=1)
                        cum_inf_norm += torch.linalg.norm(p, ord=torch.inf)
                    log_dict.update({
                        "train cumulative l2 norm": cum_l2_norm.cpu().item(),
                        "train cumulative l1 norm": cum_l1_norm.cpu().item(),
                        "train cumulative linf norm": cum_inf_norm.cpu().item(),
                    })
                wandb.log(log_dict)
            else:
                print(f"Epoch {epoch}: train loss {loss.item()}, train accuracy {(y_hat[:, -2, :].argmax(dim=1) == y[:, -2]).float().mean().item()}")
    # eval
    model.eval()
    with torch.no_grad():
        loss, accuracy = 0, 0
        for batch in val_dataloader: # only one batch
            X, y = batch['text'], batch['target']
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_hat, _, _ = model(X)
            loss += criterion(y_hat[:, -2, :], y[:, -2]).item()
            accuracy += (y_hat[:, -2, :].argmax(dim=1) == y[:, -2]).float().mean().item()
        if LOG:
            log_dict = {
                "Loss/val": loss / len(val_dataloader),
                "epoch": epoch,
                "Accuracy/val": accuracy / len(val_dataloader),
            }
            if accuracy / len(val_dataloader) > 0.98 and steps_to_98_val == -1:
                steps_to_98_val = epoch * len(train_dataloader)
                log_dict.update({
                    "Time to >98% Validation Accuracy": epoch,
                    "steps_{98%\_val} - steps_{\_train}": steps_to_98_val - step_perf_train_acc,
                })
            if accuracy/len(val_dataloader) == 1 and steps_to_100_val == -1:
                steps_to_100_val = epoch * len(train_dataloader)
                log_dict.update({
                    "perf_{98%\_val} - steps_{\_train}":  steps_to_100_val - step_perf_train_acc,
                })
            wandb.log(log_dict)
        else:
            print(f"Epoch {epoch}: test loss {loss / len(val_dataloader)}, test accuracy {accuracy / len(val_dataloader)}")

    # save model
    torch.save(model.state_dict(), "model.pt")
