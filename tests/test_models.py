import unittest
import torch
from adVAE.models.gene_expression_vae import GeneExpressionVAE
from adVAE.models.mri_vae import MRIVAE


class TestModelForward(unittest.TestCase):

    def test_gene_expression_vae_forward(self):
        model = GeneExpressionVAE(input_dim=13, hidden_dim=64, latent_dim=32)
        x = torch.randn(8, 13)
        recon, mu, log_var = model(x)

        self.assertEqual(recon.shape, (8, 13))
        self.assertEqual(mu.shape, (8, 32))
        self.assertEqual(log_var.shape, (8, 32))

    def test_mri_vae_forward(self):
        model = MRIVAE(latent_dim=64)
        x = torch.randn(4, 3, 128, 128)
        recon, mu, log_var = model(x)
        
        self.assertEqual(recon.shape, (4, 3, 128, 128))
        self.assertEqual(mu.shape, (4, 64))
        self.assertEqual(log_var.shape, (4, 64))
