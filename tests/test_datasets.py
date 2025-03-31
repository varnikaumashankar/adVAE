import unittest
import os
import torch
from adVAE.data_preprocessing.gene_expression.dataset import GeneExpressionDataset
from adVAE.data_preprocessing.mri.dataset import MRIDataset


class TestDatasets(unittest.TestCase):

    def test_gene_expression_dataset(self):
        path = "data/processed/gene_expression.pt"
        self.assertTrue(os.path.exists(path), f"Dataset not found at {path}")
        
        ds = GeneExpressionDataset(path)
        sample = ds[0]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertEqual(sample.shape[0], 13)

    def test_mri_dataset(self):
        path = "data/processed/mri.pt"
        self.assertTrue(os.path.exists(path), f"Dataset not found at {path}")

        ds = MRIDataset(path)
        sample = ds[0]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertEqual(sample.shape, (3, 128, 128))