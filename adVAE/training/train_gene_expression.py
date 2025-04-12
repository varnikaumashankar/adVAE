import os
import pickle
import torch
from torch import optim
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from datetime import datetime
from adVAE.models.gene_expression_vae import GeneExpressionVAE
from adVAE.data_preprocessing.gene_expression.dataset import GeneExpressionDataset
from adVAE.metrics.vae_loss import vae_loss
from adVAE.metrics.performance import reconstruction_accuracy
from adVAE.visualization.plot_utils import plot_loss_curve, plot_accuracy_curve


def train_model(cfg):
    """
    Train a VAE model on PCA-reduced gene expression data.

    Args:
        cfg (dict): Configuration dictionary containing training parameters and paths.
    
    Returns:
        GeneExpressionVAE: Trained VAE model.
    """
    with open(cfg["pca_model_path"], "rb") as f:
        pca_model = pickle.load(f)

    input_dim = pca_model.n_components_

    timestamp = datetime.now().strftime("%Y-%m-%d")

    dataset = GeneExpressionDataset(cfg["data_path"])
    dataloader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeneExpressionVAE(input_dim, cfg["hidden_dim"], cfg["latent_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    loss_history = []
    accuracy_history = []

    model.train()
    for epoch in range(cfg["epochs"]):
        total_loss = 0
        total_acc = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, log_var = model(batch)
            loss = vae_loss(recon, batch, mu, log_var, beta=cfg["beta"])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += reconstruction_accuracy(batch, recon, threshold=0.1)

        avg_loss = total_loss / len(dataloader.dataset)
        avg_acc = total_acc / len(dataloader)

        loss_history.append(avg_loss)
        accuracy_history.append(avg_acc)

        print(f"Epoch {epoch + 1}/{cfg['epochs']} - Loss: {avg_loss:.4f} - Accuracy: {avg_acc:.4f}")

    # Save model and plots
    os.makedirs(os.path.dirname(cfg["weights_path"]), exist_ok=True)
    torch.save(model.state_dict(), cfg["weights_path"])
    print(f"Model weights saved to {cfg['weights_path']}")

    plot_loss_curve(loss_history, save_path=os.path.join(cfg["output_dir"], f"loss_curve_{timestamp}.png"))
    plot_accuracy_curve(accuracy_history, save_path=os.path.join(cfg["output_dir"], f"accuracy_curve_{timestamp}.png"))

    return model

if __name__ == "__main__":
    from adVAE.config import gene_expression as cfg
    trained_model = train_model(cfg)