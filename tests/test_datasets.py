import unittest
import os
import torch
from datetime import datetime
from adVAE.data_preprocessing.gene_expression.dataset import GeneExpressionDataset
from adVAE.data_preprocessing.mri.dataset import MRIDataset

class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d")
        self.gene_path = f"data/processed/X_pca_{self.timestamp}.npy"
        self.mri_path = f"data/processed/mri_{self.timestamp}.pt"

    def test_gene_expression_dataset(self):
        self.assertTrue(os.path.exists(self.gene_path), f"Gene expression dataset not found at {self.gene_path}")
        
        ds = GeneExpressionDataset(self.gene_path)
        sample = ds[0]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertGreater(sample.shape[0], 0) 

    def test_mri_dataset(self):
        self.assertTrue(os.path.exists(self.mri_path), f"MRI dataset not found at {self.mri_path}")

        ds = MRIDataset(self.mri_path)
        sample = ds[0]
        self.assertIsInstance(sample, torch.Tensor)
        self.assertEqual(sample.shape[0], 1)  
        self.assertEqual(sample.shape[1:], (128, 128))  # Height/width from transform

if __name__ == "__main__":
    unittest.main()
