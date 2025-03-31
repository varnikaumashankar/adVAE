import torch
from torch.utils.data import Dataset
import os

class MRIDataset(Dataset):
    """
    Dataset class for loading preprocessed MRI tensors.
    Each sample is expected to be a 3-channel tensor of shape (3, 128, 128).
    """
    def __init__(self, data_path):
        """
        Args:
            data_path (str): Path to the .pt file containing MRI tensors.
        """
        self.data_path = data_path
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] 
