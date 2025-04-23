import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from adVAE.models.gene_expression_vae import GeneExpressionVAE
from adVAE.models.mri_vae import MRIVAE
from adVAE.metrics.vae_loss import vae_loss


class TestTraining(unittest.TestCase):

    def test_train_step_gene_expression(self):
        model = GeneExpressionVAE(input_dim=13, hidden_dim=64, latent_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(16, 13)
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        model.train()
        for batch in loader:
            batch = batch[0] 
            recon, mu, log_var = model(batch)
            loss = vae_loss(recon, batch, mu, log_var)
            loss.backward()
            optimizer.step()
            self.assertGreater(loss.item(), 0)
            break

    def test_train_step_mri(self):
        model = MRIVAE(latent_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(8, 1, 128, 128) 
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)

        model.train()
        for batch in loader:
            batch = batch[0]
            recon, mu, log_var = model(batch)
            loss = vae_loss(recon, batch, mu, log_var)
            loss.backward()
            optimizer.step()
            self.assertGreater(loss.item(), 0)
            break

if __name__ == "__main__":
    unittest.main()
