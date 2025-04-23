import os
import torch
import pandas as pd
import nibabel as nib
import numpy as np
from torchvision import transforms
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    #transforms.RandomRotation(10),  # (±10 degrees) Added for data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize to [-1, 1] as the model uses tanh activation
])

def load_metadata(csv_file):
    """
    Load metadata and filter Alzheimer’s patients.

    Args:
        csv_file (str): Path to the metadata CSV file.

    Returns:
        alzheimers_ids (list): List of Alzheimer’s patient IDs.
    """
    df = pd.read_excel(csv_file, sheet_name=0)
    alzheimers_ids = df[df['CDR'] >= 0.5]['ID'].astype(str).tolist()
    return alzheimers_ids


def extract_slices_from_img(img_path, slice_range=(80, 100)):
    """
    Extract slices from a 3D MRI image.

    Args:
        img_path (str): Path to the 3D MRI image file.
        slice_range (tuple): Range of slices to extract (start, end).
    
    Returns:
        slices (list): List of 2D slices as tensors.
    """
    img = nib.load(img_path)
    data = img.get_fdata()
    slices = []
    for z in range(*slice_range):
        if z >= data.shape[2]:
            break
        slice_2d = data[:, :, z]
        slice_2d = np.nan_to_num(slice_2d)  # Remove NaNs if any
        slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-5)
        tensor = transform((slice_norm * 255).astype(np.uint8))  # Shape: [1, 128, 128]
        slices.append(tensor)
    return slices


def preprocess_pipeline(load_example_data=False):
    """
    Preprocess the MRI data by loading metadata, extracting slices, and saving them.
    This function loads the metadata from a CSV file, filters for Alzheimer’s patients,
    extracts slices from their MRI images, and saves the slices as tensors.

    Args:
        None

    Returns:
        None
    """
    data = []

    if load_example_data:
        csv_file = "../data/OASIS_1_Info/oasis_cross-sectional-5708aa0a98d82080.xlsx"
        alzheimers_patients = load_metadata(csv_file)
        data_folder = "../data/OASIS_1"
        data_path = "../data/example_mri/processed/example_mri.pt"

        disc_nums = [1] # Only check disc1 when load_example_data is True
    else:
        csv_file = "data/OASIS_1_Info/oasis_cross-sectional-5708aa0a98d82080.xlsx"
        alzheimers_patients = load_metadata(csv_file)
        data_folder = "data/OASIS_1"
        data_path = f"data/processed/mri_{timestamp}.pt"
        disc_nums = range(1, 13)

    for pid in alzheimers_patients:
        for disc_num in disc_nums:
            img_path = os.path.join(
                data_folder, f"disc{disc_num}", pid,
                "PROCESSED/MPRAGE/T88_111",
                f"{pid}_mpr_n4_anon_111_t88_gfc.img"
            )
            if os.path.exists(img_path):
                slices = extract_slices_from_img(img_path)
                data.extend(slices)
                print(f"{pid}: extracted {len(slices)} slices")
                break
    
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    torch.save(data, data_path)
    print(f"\nSaved {len(data)} slices to {data_path}")

if __name__ == "__main__":
    preprocess_pipeline()
