import torch
import numpy as np 

def remove_train(train_set, perc_train):
    # keep a random subset of the training data according to perc_train
    rand_perm = torch.randperm(len(train_set))
    new_data = train_set.data[rand_perm][:int(perc_train*len(train_set))]
    new_targets = list(np.array(train_set.targets)[rand_perm][:int(perc_train*len(train_set))])
    train_set.data = new_data
    train_set.targets = new_targets
    return train_set

def apply_label_noise(y, label_noise, num_classes):
    # randomely change the labels of label_noise% of the training data
    num_corrupt = int(label_noise*len(y))
    rand_perm = torch.randperm(len(y))
    array_y = np.array(y)
    array_y[list(rand_perm[:num_corrupt])] = torch.randint(0, num_classes, (int(label_noise*len(y)),))
    return list(array_y)