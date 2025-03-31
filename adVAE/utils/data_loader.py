import torch
from torch.utils.data import DataLoader
from adVAE.data_preprocessing.gene_expression.dataset import GeneExpressionDataset
from adVAE.data_preprocessing.mri.dataset import MRIDataset

def load_gene_expression_data(data_path, batch_size=64, shuffle=True):
    """
    Loads preprocessed gene expression data as a DataLoader.

    Args:
        data_path (str): Path to the .pt or .csv file.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = GeneExpressionDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_mri_data(data_path, batch_size=32, shuffle=True):
    """
    Loads preprocessed MRI tensor data as a DataLoader.

    Args:
        data_path (str): Path to the .pt file containing MRI data.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader for the MRI dataset.
    """
    dataset = MRIDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)