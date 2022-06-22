import torch
from torch.optim import AdamW
import torch.nn as nn
from dataset import ArithmeticDataset, ArithmeticIterator
from tqdm import tqdm
import wandb
import argparse
from model import DecoderOnlyTransformer
from datetime import datetime

parser = argparse.ArgumentParser(description="Replication of grokking behavior observed in Power et al.'s 'Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets")
parser.add_argument("--optimization-budget", default=3e5, type=int, help="Number of training steps to run")
parser.add_argument("--wandb-project", default="grokking", type=str, help="Wandb project name")
parser.add_argument("--no-logging", action="store_true", help="Disable logging to wandb")
parser.add_argument("--num-layers", default=2, type=int, help="Number of layers in the transformer")
parser.add_argument("--num-heads", default=4, type=int, help="Number of attention heads per layer")
parser.add_argument("--d-model", default=128, type=int, help="Dimension of the model")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--weight-decay", default=1e-5, type=float, help="Weight decay")
parser.add_argument("--beta1", default=0.9, type=float, help="AdamW beta1")
parser.add_argument("--beta2", default=0.98, type=float, help="AdamW beta2")
parser.add_argument("--vocab-len", default=2000, type=int, help="Transformer vocab length")
args = parser.parse_args()

OPTIMIZATION_BUDGET = args.optimization_budget
LOG = not args.no_logging
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

if LOG:
    tags = [f"d_model={args.d_model}", f"num_layers={args.num_layers}", f"num_heads={args.num_heads}"]
    name = f"dim_{args.d_model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=args.wandb_project, settings=wandb.Settings(start_method="thread"), tags=tags, name=name)

# get dataloaders
train_dataset, val_dataset = ArithmeticDataset.splits(
        train_pct=50, # careful not to use .5
        operator="/",
        operand_length=None,
    )
train_dataloader = ArithmeticIterator(
        train_dataset,
        DEVICE,
        batchsize_hint=0, # default (512)
    )
val_dataloader = ArithmeticIterator(
    val_dataset,
    DEVICE,
    batchsize_hint=-1, # just one batch
)

# get model: decoder-only transfrormer with causal attention masking; 2 layers, width 128, 4 attention heads
model = DecoderOnlyTransformer(
    n_layers=args.num_layers,
    n_heads=args.num_heads,
    d_model=args.d_model,
    vocab_len=args.vocab_len,
    device=DEVICE,
).float().to(DEVICE)

# get optimizer
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))

# get criterion
criterion = nn.CrossEntropyLoss()

# train model
steps_per_epoch = len(train_dataloader)
for epoch in tqdm(range(int(OPTIMIZATION_BUDGET / steps_per_epoch))):
    # train
    model.train()
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        X, y = batch['text'], batch['target']
        X, y = X.to(DEVICE), y.to(DEVICE)
        y_hat = model(X)
        loss = criterion(y_hat[:, -2, :], y[:, -2])
        loss.backward()
        optimizer.step()
        if LOG:
            wandb.log({
                "Loss/train": loss.item(), 
                "epoch": epoch,
                "Accuracy/train": (y_hat[:, -2, :].argmax(dim=1) == y[:, -2]).float().mean().item(),
            })
        else:
            print(f"Epoch {epoch}: train loss {loss.item()}, train accuracy {(y_hat[:, -2, :].argmax(dim=1) == y[:, -2]).float().mean().item()}")
    # eval
    model.eval()
    with torch.no_grad():
        loss, accuracy = 0, 0
        for batch in val_dataloader: # only one batch
            X, y = batch['text'], batch['target']
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_hat = model(X)
            loss += criterion(y_hat[:, -2, :], y[:, -2]).item()
            accuracy += (y_hat[:, -2, :].argmax(dim=1) == y[:, -2]).float().mean().item()
        if LOG:
            wandb.log({
                "Loss/val": loss / len(val_dataloader),
                "epoch": epoch,
                "Accuracy/val": accuracy / len(val_dataloader),
            })
        else:
            print(f"Epoch {epoch}: test loss {loss / len(val_dataloader)}, test accuracy {accuracy / len(val_dataloader)}")

    # save model
    torch.save(model.state_dict(), "model.pt")