import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from adVAE.models.mri_vae import MRIVAE
from adVAE.data_preprocessing.mri.dataset import MRIDataset
from adVAE.metrics.vae_loss import vae_loss
from adVAE.metrics.performance import reconstruction_accuracy
from adVAE.visualization.plot_utils import plot_loss_curve, plot_accuracy_curve
from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d")

def train_model(data_path, latent_dim=64, batch_size=32, epochs=30, lr=1e-3, beta=0.1, output_dir="results/mri", device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Trains the MRI VAE model on the provided dataset.

    Args:
        data_path (str): Path to the dataset.
        latent_dim (int): Dimensionality of the latent space.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        beta (float): Weight for the KL divergence term.
        output_dir (str): Directory to save the model weights and plots.
        device (str): Device to use for training ('cuda' or 'cpu').
    
    Returns:
        float: The average loss over the training dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = MRIDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MRIVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    acc_history = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, log_var = model(batch)
            loss = vae_loss(recon, batch, mu, log_var, beta=beta)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += reconstruction_accuracy(batch, recon, threshold=0.1)

        avg_loss = total_loss / len(dataloader.dataset)
        avg_acc = total_acc / len(dataloader)

        loss_history.append(avg_loss)
        acc_history.append(avg_acc)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

    weights_path = os.path.join(output_dir, f"vae_weights_lat{latent_dim}_b{beta}_lr{lr}_{timestamp}.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

    plot_loss_curve(loss_history, save_path=os.path.join(output_dir, f"loss_curve_{timestamp}.png"))
    plot_accuracy_curve(acc_history, save_path=os.path.join(output_dir, f"accuracy_curve_{timestamp}.png"))

    return avg_loss

def grid_search():
    """
    Perform a grid search over hyperparameters for the MRI VAE model.
    This function will iterate over different combinations of latent dimensions, betas, and learning rates, training the model for each combination and
    saving the best performing configuration based on the average loss.

    Args:
        None

    Returns:
        None
    """
    data_path = f"data/processed/mri_{timestamp}.pt"
    latent_dims = [32, 64]
    betas = [0.1, 0.5]
    learning_rates = [1e-3, 5e-4]

    best_config = None
    best_loss = float('inf')

    for ld in latent_dims:
        for beta in betas:
            for lr in learning_rates:
                print(f"\nTraining with latent_dim={ld}, beta={beta}, lr={lr}")
                loss = train_model(data_path, latent_dim=ld, beta=beta, lr=lr, batch_size=32, epochs=30, output_dir=f"results/mri/model/ld{ld}_b{beta}_lr{lr}")
                if loss < best_loss:
                    best_loss = loss
                    best_config = (ld, beta, lr)

    print(f"\nBest config: latent_dim={best_config[0]} | beta={best_config[1]} | lr={best_config[2]} | loss {best_loss:.4f}")

if __name__ == "__main__":
    #grid_search() # Uncomment this line to run grid search after testing
    train_model(data_path=f"data/processed/mri_{timestamp}.pt", latent_dim=64, batch_size=32, epochs=50, lr=1e-3, beta=0.1) # Test with default parameters to ensure everything works