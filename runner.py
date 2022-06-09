import torch
from torch.optim import Adam
import torch.nn as nn
from model import Transformer
from dataset import (
    # DEFAULT_DATA_DIR,
    # EOS_TOKEN,
    # VALID_OPERATORS,
    ArithmeticDataset,
    ArithmeticIterator,
)
from tqdm import tqdm
import wandb

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 3e5
LOG = True

if LOG:
    wandb.init(project="grok", settings=wandb.Settings(start_method="thread"))

# get dataloaders
train_dataset, val_dataset = ArithmeticDataset.splits(
        train_pct=0.5,
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
optimizer = Adam(model.parameters(), lr=0.001)

# get criterion
criterion = nn.CrossEntropyLoss()

# train model
steps_per_epoch = len(train_dataloader)
for epoch in tqdm(range(int(NUM_EPOCHS)//steps_per_epoch)):
    # train
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        x, y = batch['text'], batch['target']
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y_hat, _, _ = model(x) # y_hat is 47x6x2000
        loss = criterion(y_hat.permute(0, 2, 1), y)
        loss.backward()
        optimizer.step()
        if LOG:
            wandb.log({
                "Loss/train": loss.item(), 
                "epoch": epoch,
                "Accuracy/train": (y_hat.argmax(dim=2) == y).float().mean().item(),
            })
        else:
            print(f"Epoch {epoch}: train loss {loss.item()}")
    # eval
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader: # only one batch
            x, y = batch['text'], batch['target']
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_hat, _, _ = model(x)
            loss = criterion(y_hat.permute(0, 2, 1), y)
            if LOG:
                wandb.log({
                    "Loss/val": loss.item(), 
                    "epoch": epoch,
                    "Accuracy/val": (y_hat.argmax(dim=2) == y).float().mean().item(),
                })
            else:
                print(f"Epoch {epoch}: test loss {loss.item()}")

    # save model
    torch.save(model.state_dict(), "model.pt")








