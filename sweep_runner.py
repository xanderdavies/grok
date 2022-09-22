import torch 
import wandb 
from multiprocessing import Process
import os
import argparse

parser = argparse.ArgumentParser(description="Replication of grokking behavior observed in Power et al.'s 'Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets")
parser.add_argument("--sweep", type=str, help="Sweep id agents to run")
arguments = parser.parse_args()

NUM_JOBS = 2

def run():
    os.system(f"wandb agent {arguments.sweep}")

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    wandb.setup(settings=wandb.Settings(start_method="thread"))

    processes = []
    for i in range(NUM_JOBS):
        processes.append(Process(target=run))

    for p in processes:
        print("starting process...")
        import time; time.sleep(1)
        p.start()

    for p in processes:
        p.join()