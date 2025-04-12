import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.decomposition import PCA
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d")

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
        pd.DataFrame: Combined dataframe containing all loaded data.
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
    return pd.concat(dfs, axis=0, ignore_index=True)

def remove_duplicates(df, id_column="ProbeID"): 
    """
    Remove duplicate entries based on an ID column.

    Args:
        df (pd.DataFrame): Input dataframe.
        id_column (str): Column to check for duplicates.

    Returns:
        df (pd.DataFrame): Dataframe with duplicates removed.
    """
    if id_column not in df.columns:
        print(f"Column '{id_column}' not found — skipping duplicate removal.")
        return df
    
    before = len(df)
    df = df.drop_duplicates(subset=id_column)
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
        scaler (StandardScaler or MinMaxScaler): Fitted scaler object.
        scaled (np.ndarray): Scaled data as numpy array.
    """
    available_cols = [col for col in vae_feature_columns if col in df.columns]
    missing_cols = set(vae_feature_columns) - set(available_cols)
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    scaled = scaler.fit_transform(df[available_cols])
    return scaler, scaled

def describe_data(df):
    """
    Generate descriptive statistics of the dataframe.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Descriptive statistics.
    """
    return df.describe()

def run_pca(scaled_data, n_components=0.95):
    """
    Perform PCA on scaled data.

    Args:
        scaled_data (pd.DataFrame): Scaled data.
        n_components (int): Number of PCA components to keep.

    Returns:
        X_pca (np.ndarray): PCA transformed data.
        pca_model (PCA): Fitted PCA model.
    """
    pca_model = PCA(n_components=n_components)
    X_pca = pca_model.fit_transform(scaled_data)
    return X_pca, pca_model

def preprocess_pipeline(n_components=0.95, data_folder="data/AMP_AD_MSBB_MSSM", scale_method = "standard", visualize = False, aggregate_by_gene = True, plot_path = None, save_dir="data/processed"):
    """
    Complete preprocessing pipeline for AMP_AD_MSBB_MSSM gene expression data.

    Args:
        n_components (int): Number of PCA components.
        data_folder (str): Path to folder containing gene expression files.
        scale_method (str): Scaling method ('standard' or 'minmax').
        visualize (bool): Whether to visualize distributions.
        aggregate_by_gene (bool): Whether to aggregate by gene.
        plot_path (str): Path to save distribution plot.
        save_dir (str): Directory to save processed data.
    
    Returns:
        df (pd.DataFrame): Processed dataframe.
        X_pca (np.ndarray): PCA transformed data.
        scaler (StandardScaler or MinMaxScaler): Fitted scaler object.
        pca_model (PCA): Fitted PCA model.
    """
    df = load_microarray_data(data_folder)

    if not aggregate_by_gene:
        df = remove_duplicates(df, id_column="ProbeID")

    df = fill_missing_values(df)
    if aggregate_by_gene:
        df = aggregate_probes_by_gene(df)

    if visualize:
        visualize_distribution(df, output_path=plot_path)

    raw_path = os.path.join(save_dir, f"raw_gene_expression_{timestamp}.csv")
    df.to_csv(raw_path, index=False)

    scaler, scaled = scale_data(df, method=scale_method)
    X_pca, pca_model = run_pca(scaled, n_components=n_components)

    np.save(os.path.join(save_dir, f"X_pca_{timestamp}.npy"), X_pca)
    with open(os.path.join(save_dir, f"pca_model_{timestamp}.pkl"), "wb") as f:
        pickle.dump(pca_model, f)
    with open(os.path.join(save_dir, f"scaler_{timestamp}.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved raw CSV to {raw_path}")
    print(f"Saved PCA-reduced data and models to {save_dir}")
    return df, X_pca, scaler, pca_model

if __name__ == "__main__":
    preprocess_pipeline(data_folder="data/AMP_AD_MSBB_MSSM")



