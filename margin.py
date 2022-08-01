from typing import List, Tuple
import torch
import matplotlib.pyplot as plt

def get_margin_lists(outputs, labels) -> Tuple[List, List]:
    """
    The margin for each datapoint is the difference btween the correct class and the highest incorrect class.
    """
    # normalize outputs 
    outputs, norm_outputs = outputs.clone(), outputs.clone()
    norm_outputs = norm_outputs / torch.norm(norm_outputs, dim=1).unsqueeze(1)
    # get the correct class
    correct_class_norm = norm_outputs[torch.arange(len(labels)), labels]
    correct_class = outputs[torch.arange(len(labels)), labels]
    # get the highest incorrect class
    norm_outputs[torch.arange(len(labels)), labels] = -1e10
    outputs[torch.arange(len(labels)), labels] = -1e10
    incorrect_classes_norm = torch.max(norm_outputs, dim=1)[0]
    incorrect_classes = torch.max(outputs, dim=1)[0]
    # get the margin
    margin_norm = correct_class_norm - incorrect_classes_norm
    margin = correct_class - incorrect_classes
    return margin, margin_norm

def investigate_param_scaling(inputs, labels, model):
    """
    Creates a graph with scalar (applied to each constant) as x axis, and 
    margin as y axis. Investigates relationship between margin and parameter scaling.
    """
    # get the margin for each constant
    avg_margins, avg_normalized_margins = [], []
    min_margins, min_normalized_margins = [], []
    max_margins, max_normalized_margins = [], []
    for sc in range(1, 50, 1):
        scalar = sc/10
        # apply the scalar to the parameters
        for param in model.parameters():
            param.data *= scalar
        # get the margin
        margin, margin_norm = get_margin_lists(model(inputs)[:, -2, :], labels[:, -2])
        avg_margins.append(torch.mean(torch.tensor(margin)).item())
        avg_normalized_margins.append(torch.mean(torch.tensor(margin_norm)).item())
        min_margins.append(torch.min(torch.tensor(margin)).item())
        min_normalized_margins.append(torch.min(torch.tensor(margin_norm)).item())
        max_margins.append(torch.max(torch.tensor(margin)).item())
        max_normalized_margins.append(torch.max(torch.tensor(margin_norm)).item())
        
        for param in model.parameters():
            param.data /= scalar

        # make a figure with 3 plots (one for avg, one for min, one for max), where each plot has a line for margin and normalized margin.
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(avg_margins, label="avg margin")
        axs[0].plot(avg_normalized_margins, label="avg normalized margin")
        axs[0].set_xlabel("scalar")
        axs[0].set_ylabel("margin")
        axs[0].legend()
        axs[1].plot(min_margins, label="min margin")
        axs[1].plot(min_normalized_margins, label="min normalized margin")
        axs[1].set_xlabel("scalar")
        axs[1].set_ylabel("margin")
        axs[1].legend()
        axs[2].plot(max_margins, label="max margin")
        axs[2].plot(max_normalized_margins, label="max normalized margin")
        axs[2].set_xlabel("scalar")
        axs[2].set_ylabel("margin")
        axs[2].legend()
        plt.savefig("scalar_vs_margin.png")

        # figure same as above, but just with the normalized margin
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(avg_normalized_margins, label="avg normalized margin")
        axs[0].set_xlabel("scalar")
        axs[0].set_ylabel("margin")
        axs[0].legend()
        axs[1].plot(min_normalized_margins, label="min normalized margin")
        axs[1].set_xlabel("scalar")
        axs[1].set_ylabel("margin")
        axs[1].legend()
        axs[2].plot(max_normalized_margins, label="max normalized margin")
        axs[2].set_xlabel("scalar")
        axs[2].set_ylabel("margin")
        axs[2].legend()
        plt.savefig("scalar_vs_normalized_margin.png")

def get_confidence(outputs, labels):
    """
    Returns the softmax confidence of the model for each datapoint, and the softmax confidenc after normalization.
    """
    conf = torch.softmax(outputs, dim=1)[torch.arange(len(labels)), labels]
    conf_norm = torch.softmax(outputs/torch.norm(outputs, dim=1).unsqueeze(1), dim=1)[torch.arange(len(labels)), labels]
    return conf, conf_norm