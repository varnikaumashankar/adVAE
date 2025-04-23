import torch
import os
from PIL import Image
import numpy as np
from adVAE.models.mri_vae import MRIVAE
from adVAE.utils.generator import generate_synthetic_data_from_posteriors
from adVAE.data_preprocessing.mri.dataset import MRIDataset
from torch.utils.data import DataLoader
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d")

def save_single_slice_as_gif(slice_tensor, path):
    """
    Save a single slice tensor as a GIF image.

    Args:  
        slice_tensor (torch.Tensor): The tensor representing the MRI slice.
        path (str): The path where the GIF will be saved.
    
    Returns:
        None
    """
    array = slice_tensor.squeeze().clamp(-1, 1)
    array = ((array + 1) / 2.0 * 255).byte().cpu().numpy()
    img = Image.fromarray(array, mode='L')
    img.save(path, save_all=False)

def generate_and_save(weights_path, latent_dim, num_samples, output_dir, data_path=f"data/processed/mri_{timestamp}.pt", batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Generate synthetic MRI images and save them as GIFs.

    Args:
        weights_path (str): Path to the trained model weights.
        latent_dim (int): Dimensionality of the latent space.
        num_samples (int): Number of synthetic samples to generate.
        output_dir (str): Directory to save the generated GIFs.
        data_path (str): Path to the dataset.
        batch_size (int): Batch size for data loading.
        device (str): Device to run computations on ('cuda' or 'cpu').
    
    Returns:
        None
    """
    model = MRIVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    dataset = MRIDataset(data_path) # Load real MRI dataset to sample from posteriors
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    synthetic = generate_synthetic_data_from_posteriors(model, dataloader, num_samples, device=device) # Generate using learned posterior z samples
    synthetic_tensor = torch.tensor(synthetic, dtype=torch.float32)

    os.makedirs(output_dir, exist_ok=True)
    for i, img_tensor in enumerate(synthetic_tensor):
        save_path = os.path.join(output_dir, f"synthetic_mri_{i+1}_{timestamp}.gif")
        save_single_slice_as_gif(img_tensor, save_path)

    print(f"Saved {num_samples} synthetic MRI GIFs to {output_dir}")

if __name__ == "__main__":
    generate_and_save(weights_path=f"results/mri/vae_weights_lat64_b0.1_lr0.001_{timestamp}.pth", latent_dim=64, num_samples=10, output_dir="results/mri/synthetic_mri_gifs")