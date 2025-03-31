import numpy as np
import torch
import torch.nn.functional as F


def reconstruction_accuracy(original, reconstructed, threshold=0.1):
    """
    Computes accuracy as the percentage of values reconstructed within a threshold.

    Args:
        original (Tensor or ndarray): Ground truth input.
        reconstructed (Tensor or ndarray): Reconstructed output.
        threshold (float): Allowed error margin.

    Returns:
        float: Accuracy between 0 and 1.
    """
    if hasattr(original, 'detach'):
        original = original.detach().cpu().numpy()
    if hasattr(reconstructed, 'detach'):
        reconstructed = reconstructed.detach().cpu().numpy()

    correct = np.abs(original - reconstructed) <= threshold
    return correct.sum() / correct.size


def mean_absolute_error(original, reconstructed):
    """
    Computes the Mean Absolute Error (MAE) between original and reconstructed data.

    Args:
        original (Tensor or ndarray): Ground truth input.
        reconstructed (Tensor or ndarray): Reconstructed output.

    Returns:
        float: Mean absolute error.
    """
    if hasattr(original, 'detach'):
        original = original.detach().cpu().numpy()
    if hasattr(reconstructed, 'detach'):
        reconstructed = reconstructed.detach().cpu().numpy()

    return np.mean(np.abs(original - reconstructed))


