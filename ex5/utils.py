import torch
import numpy as np
import random
import matplotlib.pyplot as plt

BASELINE = 0.6

def plot_training_log(training_log:dict, test_accuracy:float=None, figsize=(10, 4), show_baseline=True):

    baseline = BASELINE

    plt.figure(figsize=figsize, dpi=100)

    save_epoch_color = "green"
    test_acc_color = "magenta"
    baseline_color = "red"

    plt.subplot(1, 2, 1)

    plt.plot(training_log["train_acc"], label="Training")
    plt.plot(training_log["val_acc"], label="Validation")
    plt.title("Accuracy")

    plt.vlines(training_log["best_val_epoch"], 0, 1, 
               linestyles="dashed", colors=save_epoch_color, label="Save Epoch")

    if test_accuracy is not None:
        plt.hlines(test_accuracy, 0, len(training_log["train_acc"]), 
                   linestyles="dashed", colors=test_acc_color, label="Test")

    if show_baseline:
        plt.hlines(baseline, 0, len(training_log["train_acc"]), 
                   linestyles="dashed", colors=baseline_color, label="Baseline")

    plt.xlabel("Epoch")
    plt.legend()
    plt.grid()

    max_vline = max(max(training_log["val_loss"][1:]), max(training_log["train_loss"][1:])) + 0.1
    min_vline = min(min(training_log["val_loss"][1:]), min(training_log["train_loss"][1:])) - 0.1

    plt.subplot(1, 2, 2)
    plt.plot(training_log["train_loss"], label="Training")
    plt.plot(training_log["val_loss"], label="Validation")
    plt.vlines(training_log["best_val_epoch"], min_vline, max_vline, 
               linestyles="dashed", colors=save_epoch_color, label="Save Epoch")
    plt.title("Loss")

    plt.xlabel("Epoch")

    plt.legend()
    plt.grid()
    plt.show()

def seed_everything(seed:int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
