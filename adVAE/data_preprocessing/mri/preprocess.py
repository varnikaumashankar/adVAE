import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((128, 128)),
                                transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])

def load_metadata(csv_file):
    """
    Load metadata and filter Alzheimer’s patients.
    """
    df = pd.read_excel(csv_file, sheet_name=0)
    alzheimers_ids = df[df['CDR'] >= 0.5]['ID'].astype(str).tolist()
    return alzheimers_ids

def load_patient_images(patient_id, data_dir, max_disc=12):
    """
    Loads and preprocesses MRI slices for a patient.
    """
    images = []
    for disc_num in range(1, max_disc + 1):
        disc_path = os.path.join(data_dir, f"disc{disc_num}")
        img_paths = [
            os.path.join(disc_path, patient_id, "PROCESSED/MPRAGE/T88_111", f"{patient_id}_mpr_n4_anon_111_t88_gfc_tra_90.gif"),
            os.path.join(disc_path, patient_id, "PROCESSED/MPRAGE/T88_111", f"{patient_id}_mpr_n4_anon_111_t88_gfc_sag_95.gif"),
            os.path.join(disc_path, patient_id, "PROCESSED/MPRAGE/T88_111", f"{patient_id}_mpr_n4_anon_111_t88_gfc_cor_110.gif")
        ]
        for path in img_paths:
            if os.path.exists(path):
                img = Image.open(path)
                img_tensor = transform(img)
                images.append(img_tensor)
        if len(images) == 3:
            return torch.cat(images, dim=0)  # Shape: (3, 128, 128)
    return None


def preprocess_pipeline(use_example_data=False):
    """
    Preprocess all Alzheimer's MRI patients and save the dataset.
    """
    if use_example_data:
        print("Using example MRI data...")
        data_dir = "../data/example_mri"
        csv_file = "../data/Example_OASIS_1_Info/oasis_cross-sectional-5708aa0a98d82080.xlsx"
        save_path = "../data/processed/example_mri.pt"

        alzheimers_patients = load_metadata(csv_file)
        data = []
        for pid in alzheimers_patients:
            img_tensor = load_patient_images(pid, data_dir, max_disc=1)
            if img_tensor is not None:
                data.append(img_tensor)
                print(f"Loaded {pid}: shape {img_tensor.shape}")
            else:
                print(f"Skipping {pid} — insufficient slices found.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(data, save_path)
        print(f"\nSaved {len(data)} example MRI volumes to {save_path}")
        return

    data_dir = "data/OASIS_1"
    csv_file = "data/OASIS_1_Info/oasis_cross-sectional-5708aa0a98d82080.xlsx"
    save_path = "data/processed/mri.pt"

    alzheimers_patients = load_metadata(csv_file)
    data = []
    for pid in alzheimers_patients:
        img_tensor = load_patient_images(pid, data_dir, max_disc=1 if use_example_data else 12)
        if img_tensor is not None:
            data.append(img_tensor)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path) 
    print(f"Saved {len(data)} patients' MRI tensors to {save_path}")

