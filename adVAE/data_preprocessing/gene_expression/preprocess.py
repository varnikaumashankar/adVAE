import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Hardcoding for AMP_AD_MSBB_MSSM gene expression files

vae_feature_columns = ['k.all', 'k.in', 'k.out', 'k.diff', 'k.in.normed', 'k.all.normed', 'to.all',
                        'to.in', 'to.out', 'to.diff', 'to.in.normed', 'to.all.normed', 'standard.deviation']

metadata_columns = ['ProbeID', 'AccessionID', 'GeneSymbol', 'EntrezGeneID']

def load_microarray_data(data_folder="data/AMP_AD_MSBB_MSSM", file_types=(".csv", ".tsv")):
    """
    Load and concatenate all gene expression files in a folder.

    Args:
        data_folder (str): Path to folder containing .csv or .tsv files.
        file_types (tuple): Allowed file extensions. Default is (".csv", ".tsv").

    Returns:
        combined_df (pd.DataFrame): Combined dataframe containing all loaded data.
    """
    dfs = []
    for filename in os.listdir(data_folder):
        if filename.endswith(file_types):
            if filename.endswith('.tsv'):
                sep = '\t'  
            else:
                sep = ','
            df = pd.read_csv(os.path.join(data_folder, filename), sep=sep)
            dfs.append(df)
    combined_df = pd.concat(dfs, axis=0, ignore_index=True)
    return combined_df

def remove_duplicates(df, id_column="ProbeID"): 
    """
    Remove duplicate entries based on an ID column.

    Args:
        df (pd.DataFrame): Input dataframe.
        id_column (str): Column to check for duplicates.

    Returns:
        df (pd.DataFrame): Dataframe with duplicates removed.
    """
    before = len(df)
    df = df.drop_duplicates(subset=id_column, errors='ignore')
    after = len(df)
    print(f"Removed {before - after} duplicate entries.")
    return df

def fill_missing_values(df):
    """
    Fill missing or null numeric values in the dataframe using column-wise means.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        df (pd.DataFrame): Dataframe with missing values filled.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    missing_before = df[numeric_cols].isnull().sum().sum()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    missing_after = df[numeric_cols].isnull().sum().sum()
    print(f"Missing values filled: {missing_before - missing_after}")
    return df

def aggregate_probes_by_gene(df, id_column="EntrezGeneID"):
    """
    Aggregate probe-level features into gene-level features using mean.

    Args:
        df (pd.DataFrame): Input dataframe.
        id_column (str): Column representing gene IDs.

    Returns:
        grouped_df (pd.DataFrame): Aggregated dataframe with one row per gene.
    """
    if id_column not in df.columns:
        print(f"Cannot aggregate: '{id_column}' column not found.")
        return df
    numeric_df = df[vae_feature_columns + [id_column]].dropna()
    grouped_df = numeric_df.groupby(id_column)[vae_feature_columns].mean().reset_index()
    print(f"Aggregated probe-level data to {len(grouped_df)} unique genes.")
    return grouped_df

def visualize_distribution(df, output_path=None):
    """
    Plot boxplots of all numeric columns to visualize feature distributions.

    Args:
        df (pd.DataFrame): Input dataframe.
        output_path (str, optional): If provided, saves plot to this path.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df[numeric_cols])
    plt.xticks(rotation=90)
    plt.title("Distribution of Numerical Features")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"Distribution plot saved to {output_path}")
    else:
        plt.show()

def scale_data(df, method="standard"):
    """
    Scale numerical features using StandardScaler or MinMaxScaler.

    Args:
        df (pd.DataFrame): Input dataframe.
        method (str): 'standard' or 'minmax' (default: 'standard').

    Returns:
        df_scaled (pd.DataFrame): Scaled dataframe.
    """
    available_cols = [col for col in vae_feature_columns if col in df.columns]
    missing_cols = set(vae_feature_columns) - set(available_cols)
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    scaled = scaler.fit_transform(df[available_cols])
    df_scaled = pd.DataFrame(scaled, columns=available_cols)
    return df_scaled

def describe_data(df):
    """
    Generate descriptive statistics of the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Descriptive statistics.
    """
    return df.describe()

def save_processed_data(df, save_path="gene_expression.pt"):
    """
    Save processed data to a .pt or .csv file.

    Args:
        df (pd.DataFrame): Preprocessed data.
        save_path (str): Path to save file (.pt or .csv).
    """
    if save_path.endswith(".pt"):
        torch.save(torch.tensor(df.values, dtype=torch.float32), save_path)
    elif save_path.endswith(".csv"):
        df.to_csv(save_path, index=False)
    else:
        raise ValueError("Unsupported file type. Use '.pt' or '.csv'.")
    print(f"[INFO] Saved processed data to {save_path}")

def preprocess_pipeline(data_folder="data/AMP_AD_MSBB_MSSM", scale_method = "standard", visualize = False, aggregate_by_gene = False, return_stats = False, plot_path = None, return_metadata = False, save_path = False):
    """
    Complete preprocessing pipeline for AMP_AD_MSBB_MSSM gene expression data.

    Args:
        data_folder (str): Path to folder containing expression data files.
        scale_method (str): 'standard' or 'minmax'. Defaults to 'standard'.
        visualize (bool): If True, displays or saves boxplot.
        aggregate_by_gene (bool): If True, aggregates multiple probes to one gene.
        return_stats (bool): If True, returns summary statistics.
        plot_path (str): Optional path to save boxplot image.
        return_metadata (bool): If True, returns metadata columns.

    Returns:
        df_scaled (pd.DataFrame): Scaled data
        describe_data(df_scaled) (pd.DataFrame) (optional): Descriptive stats
        metadata (pd.DataFrame) (optional): Metadata
    """
    
    df = load_microarray_data(data_folder)

    if not aggregate_by_gene:
        df = remove_duplicates(df, id_column="ProbeID")

    df = fill_missing_values(df)
    if aggregate_by_gene:
        df = aggregate_probes_by_gene(df)

    if visualize:
        visualize_distribution(df, output_path=plot_path)

    df_scaled = scale_data(df, method=scale_method)
    if return_metadata:
        metadata = df[metadata_columns].copy() if all(col in df.columns for col in metadata_columns) else None
        if metadata is None:
            print("One or more metadata columns not found. Returning None for metadata.")
        if return_stats:
            return df_scaled, describe_data(df_scaled), metadata
        return df_scaled, metadata

    if return_metadata:
        metadata = df[metadata_columns].copy() if all(col in df.columns for col in metadata_columns) else None
        if metadata is None:
            print("One or more metadata columns not found. Returning None for metadata.")
        if save_path:
            if not os.path.isabs(save_path):
                os.makedirs("data/processed", exist_ok=True)
                save_path = os.path.join("data/processed", save_path)
            save_processed_data(df_scaled, save_path)
        if return_stats:
            return df_scaled, describe_data(df_scaled), metadata
        return df_scaled, metadata

    if return_stats:
        if save_path:
            if not os.path.isabs(save_path):
                os.makedirs("data/processed", exist_ok=True)
                save_path = os.path.join("data/processed", save_path)
            save_processed_data(df_scaled, save_path)
        return df_scaled, describe_data(df_scaled)
    return df_scaled


