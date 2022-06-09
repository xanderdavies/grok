import torch
from torch.optim import AdamW
import torch.nn as nn
from model import Transformer
from dataset import ArithmeticDataset, ArithmeticIterator
from tqdm import tqdm
import wandb

OPTIMIZATION_BUDGET = 3e5
LOG = True
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

if LOG:
    wandb.init(project="grok", settings=wandb.Settings(start_method="thread"))

# get dataloaders
train_dataset, val_dataset = ArithmeticDataset.splits(
        train_pct=50, # careful not to use .5
        operator="/",
        operand_length=None,
        data_dir="data",
    )
train_dataloader = ArithmeticIterator(
        train_dataset,
        DEVICE,
        batchsize_hint=0,
    )
val_dataloader = ArithmeticIterator(
    val_dataset,
    DEVICE,
    batchsize_hint=-1,
)

# get model: decoder-only transfrormer with causal attention masking; 2 layers, width 128, 4 attention heads
model = Transformer(
    n_layers=2,
    n_heads=4,
    d_model=128,
)
model.to(DEVICE)
model.train()

# get optimizer
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5, betas=(0.9, 0.98))

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
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        y_hat, _, _ = model(X) # y_hat is (batch_size, seq_len, vocab_size)
        # only calculate the loss on the answer part of the equation
        loss = criterion(y_hat[:, -2, :], y[:, -2])
        loss.backward()
        optimizer.step()
        if LOG:
            wandb.log({
                "Loss/train": loss.item(), 
                "epoch": epoch,
                "Accuracy/train": ((y_hat[:, -2, :]).argmax(dim=1) == y[:, -2]).float().mean().item(),
            })
        else:
            print(f"Epoch {epoch}: train loss {loss.item()}, train accuracy {((y_hat[:, -2, :]).argmax(dim=1) == y[:, -2]).float().mean().item()}")
    # eval
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader: # only one batch
            X, y = batch['text'], batch['target']
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_hat, _, _ = model(X)
            loss = criterion(y_hat[:, -2, :], y[:, -2])
            if LOG:
                wandb.log({
                    "Loss/val": loss.item(), 
                    "epoch": epoch,
                    "Accuracy/val": ((y_hat[:, -2, :]).argmax(dim=1) == y[:, -2]).float().mean().item(),
                })
            else:
                print(f"Epoch {epoch}: test loss {loss.item()}, test accuracy {((y_hat[:, -2, :]).argmax(dim=1) == y[:, -2]).float().mean().item()}")

    # save model
    torch.save(model.state_dict(), "model.pt")








