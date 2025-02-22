import unittest
import torch
from adVAE.models.vae import VAE

class TestVAE(unittest.TestCase):
    def test_forward_pass(self):
        model = VAE(input_dim=13, hidden_dim=64, latent_dim=10)
        x = torch.randn(5, 13)
        recon_x, mu, log_var = model(x)
        self.assertEqual(recon_x.shape, x.shape)
        self.assertEqual(mu.shape, (5, 10))
        self.assertEqual(log_var.shape, (5, 10))

    def test_reparameterize(self):
        model = VAE(13, 64, 10)
        mu = torch.zeros(5, 10)
        log_var = torch.zeros(5, 10)
        z = model.reparameterize(mu, log_var)
        self.assertEqual(z.shape, mu.shape)