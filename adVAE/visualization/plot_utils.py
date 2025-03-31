import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss_curve(loss_history, save_path=None, title="Training Loss"):
    plt.figure()
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")
    else:
        plt.show()


def plot_reconstruction_distribution(original, reconstructed, save_path=None, title_prefix=""):
    """
    Plots the original and reconstructed samples side by side for comparison.

    Args:
        original (np.ndarray or torch.Tensor): Original input data.
        reconstructed (np.ndarray or torch.Tensor): Reconstructed data.
        save_path (str, optional): Path to save the plot.
        title_prefix (str): Optional prefix for plot title.

    Returns:
        None
    """
    original = original.cpu().numpy() if hasattr(original, 'cpu') else original
    reconstructed = reconstructed.cpu().numpy() if hasattr(reconstructed, 'cpu') else reconstructed

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(original.flatten(), bins=50, alpha=0.7, color='blue', label='Original')
    plt.title(f"{title_prefix} Original Distribution")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(reconstructed.flatten(), bins=50, alpha=0.7, color='green', label='Reconstructed')
    plt.title(f"{title_prefix} Reconstructed Distribution")
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Reconstruction distribution plot saved to {save_path}")
    else:
        plt.show()


def plot_accuracy_curve(accuracy_history, save_path=None, title="Accuracy Curve"):
    plt.figure()
    plt.plot(range(1, len(accuracy_history)+1), accuracy_history, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Accuracy curve saved to {save_path}")
    else:
        plt.show()
