# This replicates epoch-wise double descent, as observed in ["Deep Double Descent: Where Bigger Models and More Data Hurt"](https://arxiv.org/abs/1912.02292).

import torch
import torch.nn as nn 
from torch.optim import Adam, AdamW, SGD
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, Normalize

from tqdm import tqdm
import wandb
from multiprocessing import Process
from datetime import datetime

from resnet_18k import make_resnet18k
from utils import remove_train, apply_label_noise
import argparse

"""
Hyperparameters
"""

# parser = argparse.ArgumentParser(description="Double Descent")
# parser.add_argument("--dset-size", type=float, required=True)
# arguments = parser.parse_args() 

# PERC_TRAIN = arguments.dset_size
PERC_TRAIN = [0.1, 1]
# PERC_TRAIN = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1] # [1, 0.8, 0.6, 0.4] #.001 # how much of the training data to use (1 = all)

WIDTH_PARAM = 64 # [3, 12, 64] are in DDD
LR =  0.0001 # 0.01 # per DDD, but fails?
LABEL_NOISE = .15 # per DDD
EPOCHS = 4000 # per DDD
BATCH_SIZE = 128 # per DDD

WEIGHT_DECAY = 0 # per DDD, could try 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'; print("using:", DEVICE)
OPT = "Adam" # per DDD

RANDOM_SEED = True
if not RANDOM_SEED:
    torch.random.manual_seed(0)
NUM_JOBS = 2
LOAD_PATH = None

args = {
    "Opt": OPT, 
    "Weight Decay": WEIGHT_DECAY,
    "Label Noise": LABEL_NOISE,
    "Percent Training Data": PERC_TRAIN, 
    "Learning Rate": LR, 
    "Resnet-18 Width Parameter": WIDTH_PARAM,
    "Random Seed": RANDOM_SEED
}

def run(perc_train):
    """
    Data
    """
    transform = Compose([
        RandomCrop(32, padding=4), # per DDD
        RandomHorizontalFlip(), # per DDD
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_set = CIFAR10('./data', train=True, download=True, transform=transform)
    test_set = CIFAR10('./data', train=False, download=True, transform=transform)
    train_set = remove_train(train_set, perc_train)
    train_set.targets = apply_label_noise(train_set.targets, LABEL_NOISE, len(train_set.classes))

    print(f"Train set includes {len(train_set.data)} images")
    BATCH_SIZE_UPDATED = min(len(train_set.data), BATCH_SIZE)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE_UPDATED, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE_UPDATED, shuffle=False)

    """
    Model
    """
    model = make_resnet18k(WIDTH_PARAM, num_classes=10).to(DEVICE)
    if LOAD_PATH:
        model.load_state_dict(torch.load(LOAD_PATH))
    criterion = nn.CrossEntropyLoss()
    if OPT == "Adam":
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPT == "SGD":
        # optimizer = SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        optimizer = SGD([
            {"params": model.conv1.parameters(), "lr": LR}, 
            {"params": model.layer1.parameters(), "lr": LR},
            {"params": model.layer2.parameters(), "lr": LR},
            {"params": model.layer3.parameters(), "lr": LR},
            {"params": model.layer4.parameters(), "lr": LR},
            {"params": model.linear.parameters(), "lr": LR},
        ], lr=LR, weight_decay=WEIGHT_DECAY)

    else:
        raise NotImplementedError()

    """
    Wandb
    """
    tags = [f"perc_train_{perc_train}", OPT, f"label_noise_{LABEL_NOISE}", f"width_{WIDTH_PARAM}", f"weight_decay_{WEIGHT_DECAY}", f"LR_{LR}"]
    args["Percent Training Data"] = perc_train
    name = '-'.join(tags) + f"-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project="deep-double-descent", tags=tags, name=name, config=args)
    wandb.watch(model)

    """
    Train
    """
    for ep in tqdm(range(EPOCHS)):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            wandb.log({
                "Loss/train": loss.item(),
                "Accuracy/train": (y_pred.argmax(dim=1) == y).float().mean().item(),
                "epoch": ep,
                "Loss/normalized_train": criterion((y_pred / torch.norm(y_pred, dim=1).unsqueeze(1)), y).item()
            })
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                total_loss += loss.item()
                total_acc += (y_pred.argmax(dim=1) == y).float().mean().item()
            wandb.log({
                "Loss/val": total_loss / len(test_loader),
                "Accuracy/val": total_acc / len(test_loader),
                "epoch": ep,
                "Loss/normalized_val": criterion((y_pred / torch.norm(y_pred, dim=1).unsqueeze(1)), y).item()
            })

        # torch.save(model.state_dict(), f"weights/{name}-LATEST-model.pt")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    wandb.setup()
    processes = []
    for i in range(NUM_JOBS):
        processes.append(Process(target=run, args=(PERC_TRAIN[i] if isinstance(PERC_TRAIN, list) else PERC_TRAIN,)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()