import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_microarray_data(data_folder):
    """
    Load multiple microarray data files from a specified folder and merge them.

    Args:
        data_folder (str): The path to the folder containing data files.
    
    Returns:
        combined_df (pd.DataFrame): The combined DataFrame containing all data
    """
    all_dfs = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv") or filename.endswith(".tsv"):
            file_path = os.path.join(data_folder, filename)
            df = pd.read_csv(file_path, sep='\t' if filename.endswith(".tsv") else ',')
            all_dfs.append(df)
    combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    return combined_df

def check_duplicates(df):
    """
    Check and remove duplicate entries based on 'ID'.

    Args:
        df (pd.DataFrame): The input DataFrame

    Returns:
        df (pd.DataFrame): The DataFrame with duplicates removed
    """
    initial_shape = df.shape
    df.drop_duplicates(subset='ID', inplace=True)
    final_shape = df.shape
    print(f"Duplicates removed: {initial_shape[0] - final_shape[0]}")
    return df

def handle_missing_values(df):
    """
    Handle missing values by filling with mean.

    Args:
        df (pd.DataFrame): The input DataFrame

    Returns:
        df (pd.DataFrame): The DataFrame with missing values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_before = df[numeric_cols].isnull().sum().sum()

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    missing_after = df[numeric_cols].isnull().sum().sum()
    print(f"Missing values handled: {missing_before - missing_after}")
    return df

def visualize_data_distribution(df):
    """
    Visualize the distribution of numerical columns.

    Args:
        df (pd.DataFrame): The input DataFrame

    Returns:
        None
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df[numeric_cols])
    plt.title("Distribution of Numerical Features")
    plt.xticks(rotation=90)
    plt.show()

def preprocess_data(df, scaling_method='standard'):
    """
    Preprocess microarray data by selecting numerical columns and scaling them.

    Args:
        df (pd.DataFrame): The input DataFrame
        scaling_method (str): The scaling method to use (standard/minmax)

    Returns:
        df_scaled (pd.DataFrame): The scaled DataFrame
    """
    numerical_cols = [
        'k.all', 'k.in', 'k.out', 'k.diff',
        'k.in.normed', 'k.all.normed',
        'to.all', 'to.in', 'to.out', 'to.diff',
        'to.in.normed', 'to.all.normed',
        'standard.deviation'
    ]

    available_cols = [col for col in numerical_cols if col in df.columns]
    missing_cols = set(numerical_cols) - set(available_cols)
    if missing_cols:
        print(f"Warning: Missing columns - {missing_cols}")

    X = df[available_cols].copy()

    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported scaling method. Use 'standard' or 'minmax'.")

    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=available_cols)

    return df_scaled

def data_stats(df):
    """
    Provide basic statistics of the processed data.

    Args:
        df (pd.DataFrame): The input DataFrame

    Returns:
        stats (pd.DataFrame): The statistics DataFrame
    """
    return df.describe()

def main():
    # Edit data path
    data_folder = "/Users/varnikaumashankar/Documents/UMich/Semester 2/BIOINF 576/adVAE/data/AMP_AD_MSBB_MSSM"  

    combined_data = load_microarray_data(data_folder) # Load data
    # print(f"Combined Data Shape: {combined_data.shape}")

    combined_data = handle_missing_values(combined_data) # Handle missing values (numeric only)

    processed_data = preprocess_data(combined_data, scaling_method='standard') # Preprocess data
    # print(f"Processed Data Shape: {processed_data.shape}")

    # stats = data_stats(processed_data) # Describe data
    # print(stats)
    return processed_data

if __name__ == "__main__":
    main()