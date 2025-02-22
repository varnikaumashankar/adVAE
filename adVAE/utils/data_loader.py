import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from adVAE.data_preprocessing.gene_exp_preprocess import main

def load_and_preprocess_data(path, batch_size=64):
    """
    Loads data from a TSV file, preprocesses it, and returns a DataLoader.

    Args:
        path (str): The file path to the .csv/.tsv file containing the data.
        batch_size (int): The number of samples per batch to load. Default is 64.

    Returns:
        DataLoader: A DataLoader object that provides an iterable over the processed data.
    """

    df = pd.read_csv(path)
    processed_df = main()  
    data = torch.tensor(processed_df.values, dtype=torch.float32)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)