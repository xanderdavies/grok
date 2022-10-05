# This replicates epoch-wise double descent, as observed in ["Deep Double Descent: Where Bigger Models and More Data Hurt"](https://arxiv.org/abs/1912.02292).

import torch
import torch.nn as nn 
from torch.optim import AdamW, SGD
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, Normalize

from tqdm import tqdm
import wandb
from datetime import datetime

from resnet_18k import make_resnet18k
from utils import remove_train, apply_label_noise

"""
Hyperparameters
"""
LOG = True
PERC_TRAIN = 1
WIDTH_PARAM = 64 # [3, 12, 64] are in DDD
LR =  0.0001 # per DDD
LABEL_NOISE = .2 # per DDD
EPOCHS = 4000 # per DDD
BATCH_SIZE = 128 # per DDD

WEIGHT_DECAY = 0 # per DDD, could try 1e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'; print("using:", DEVICE)
OPT = "Adam" # per DDD

RANDOM_SEED = False
if not RANDOM_SEED:
    torch.random.manual_seed(0)
NUM_JOBS = 1
LOAD_PATH = None

def run(args):
    perc_train = args["Percent Training Data"]
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
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE_UPDATED, shuffle=True)
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
        optimizer = SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        raise NotImplementedError()

    """
    Wandb
    """
    tags = [f"perc_train_{perc_train}", OPT, f"label_noise_{LABEL_NOISE}", f"width_{WIDTH_PARAM}", f"weight_decay_{WEIGHT_DECAY}", f"LR_{LR}"]
    args["Percent Training Data"] = perc_train
    name = '-'.join(tags) + f"-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    mode = None if args["Log"] else "disabled"
    wandb.init(project="deep-double-descent", tags=tags, name=name, config=args, mode=mode)
    wandb.watch(model)

    """
    Train
    """
    prev_20_accs = []
    for ep in tqdm(range(EPOCHS)):
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                total_loss += loss.item()
                total_acc += (y_pred.argmax(dim=1) == y).float().mean().item()

            
            prev_20_accs.append((y_pred.argmax(dim=1) == y).float().mean().item())
            if len(prev_20_accs) > 20:
                prev_20_accs.pop(0)

            wandb.log({
                "Loss/val": total_loss / len(test_loader),
                "Accuracy/val": total_acc / len(test_loader),
                "epoch": ep,
                "Loss/normalized_val": criterion((y_pred / torch.norm(y_pred, dim=1).unsqueeze(1)), y).item()
            })
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
            
        torch.save(model.state_dict(), f"weights/{name}-LATEST-model.pt")

if __name__ == '__main__':
    args = {
    "Opt": OPT, 
    "Weight Decay": WEIGHT_DECAY,
    "Label Noise": LABEL_NOISE,
    "Percent Training Data": PERC_TRAIN, 
    "Learning Rate": LR, 
    "Resnet-18 Width Parameter": WIDTH_PARAM,
    "Random Seed": RANDOM_SEED,
    "Log": LOG,
    }
    run(args)