import torch
from torch.utils.data import Dataset
import pandas as pd

class GeneExpressionDataset(Dataset):
    """
    PyTorch Dataset for preprocessed gene expression data.
    This class loads feature tensors saved as .pt or .csv files and optionally returns metadata alongside the data.
    
    Args:
        data_path (str): Path to .pt or .csv file containing scaled features.
        metadata_path (str, optional): Path to .csv file containing metadata. Default is None.
    """
    def __init__(self, data_path, metadata_path=None):
        if data_path.endswith(".pt"):
            self.data = torch.load(data_path)
        elif data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
            self.data = torch.tensor(df.values, dtype=torch.float32)
        else:
            raise ValueError("Unsupported data file type. Use .pt or .csv")

        if metadata_path:
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.metadata is not None:
            return item, self.metadata.iloc[idx]
        return item