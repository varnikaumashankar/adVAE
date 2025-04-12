import torch
from torch.utils.data import Dataset
import numpy as np

class GeneExpressionDataset(Dataset):
    """
    PyTorch Dataset for PCA-reduced gene expression data.
    Supports loading from .npy, .csv, or .pt files.

    Args:
        data_path (str): Path to .npy, .csv, or .pt file with features.

    Attributes:
        data (torch.Tensor): Tensor containing the PCA-reduced gene expression data.
    
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the sample at index `idx`.
    """
    def __init__(self, data_path):
        if data_path.endswith(".pt"):
            self.data = torch.load(data_path)
        elif data_path.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(data_path)
            self.data = torch.tensor(df.values, dtype=torch.float32)
        elif data_path.endswith(".npy"):
            self.data = torch.tensor(np.load(data_path), dtype=torch.float32)
        else:
            raise ValueError("Unsupported data file type. Use .pt, .csv, or .npy")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]