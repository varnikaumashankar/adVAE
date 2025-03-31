import unittest
import torch
from adVAE.models.gene_expression_vae import GeneExpressionVAE
from adVAE.models.mri_vae import MRIVAE
from adVAE.utils.data_loader import load_gene_expression_data
from adVAE.utils.data_loader import load_mri_data
from torch.utils.data import DataLoader


class TestUtils(unittest.TestCase):

    def test_load_gene_expression_data(self):
        dataloader = load_gene_expression_data("data/processed/gene_expression.pt", batch_size=4)
        batch = next(iter(dataloader))
        self.assertIsInstance(batch, torch.Tensor)
        self.assertEqual(batch.shape[1], 13)

    def test_load_mri_data(self):
        dataloader = load_mri_data("data/processed/mri.pt", batch_size=2)
        batch = next(iter(dataloader))
        self.assertIsInstance(batch, torch.Tensor)
        self.assertEqual(batch.shape[1:], (3, 128, 128))