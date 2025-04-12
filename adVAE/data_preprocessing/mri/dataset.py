import torch
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    """
    Dataset class for loading preprocessed grayscale MRI slices.
    Each sample is expected to be a 1-channel tensor of shape (1, 128, 128).

    Args:
        data_path (str): Path to the preprocessed MRI data file.
    
    Attributes:
        data (list): List of preprocessed MRI slices.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the sample at index `idx`.
    """
    def __init__(self, data_path):
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
