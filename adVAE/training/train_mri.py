import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from adVAE.models.mri_vae import MRIVAE
from adVAE.data_preprocessing.mri.dataset import MRIDataset
from adVAE.metrics.vae_loss import vae_loss
from adVAE.metrics.performance import reconstruction_accuracy
from adVAE.visualization.plot_utils import plot_loss_curve, plot_accuracy_curve

def train_model(data_path, latent_dim=64, batch_size=32, epochs=30, lr=1e-3,
    save_dir="results/mri/model", device="cuda" if torch.cuda.is_available() else "cpu"):
    os.makedirs(save_dir, exist_ok=True)
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
            loss = vae_loss(recon, batch, mu, log_var)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += reconstruction_accuracy(batch, recon, threshold=0.1)

        avg_loss = total_loss/len(dataloader.dataset)
        avg_acc = total_acc/len(dataloader)

        loss_history.append(avg_loss)
        acc_history.append(avg_acc)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_acc:.4f}")

    weights_path = os.path.join(save_dir, "vae_weights.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

    plot_loss_curve(loss_history, save_path=os.path.join(save_dir, "loss_curve.png"))
    plot_accuracy_curve(acc_history, save_path=os.path.join(save_dir, "accuracy_curve.png"))

    return model


if __name__ == "__main__":
    trained_model = train_model(data_path="data/processed/mri.pt", latent_dim=64, batch_size=32, epochs=30, 
                                  lr=1e-3)
