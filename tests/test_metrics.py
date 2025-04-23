import unittest
import torch
from adVAE.metrics.performance import reconstruction_accuracy, mean_absolute_error
from adVAE.metrics.vae_loss import vae_loss


class TestMetrics(unittest.TestCase):

    def test_vae_loss(self):
        x = torch.randn(10, 13)
        recon = x + 0.05 * torch.randn_like(x)
        mu = torch.randn(10, 32)
        log_var = torch.randn(10, 32)
        loss = vae_loss(recon, x, mu, log_var)
        self.assertGreater(loss.item(), 0)

    def test_reconstruction_accuracy(self):
        x = torch.randn(5, 1, 128, 128)
        recon = x + 0.05 * torch.randn_like(x)
        acc = reconstruction_accuracy(x, recon, threshold=0.1)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)

    def test_mean_absolute_error(self):
        x = torch.randn(5, 1, 128, 128)
        recon = x + 0.05 * torch.randn_like(x)
        mae = mean_absolute_error(x, recon)
        self.assertGreaterEqual(mae, 0)

if __name__ == "__main__":
    unittest.main()
