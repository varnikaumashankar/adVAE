import unittest
import pandas as pd
import torch
import os
from adVAE.data_preprocessing.gene_expression.preprocess import preprocess_pipeline
from adVAE.data_preprocessing.mri.preprocess import preprocess_pipeline


class TestPreprocessing(unittest.TestCase):
    def test_gene_expression_preprocessing(self):
        path = "data/AMP_AD_MSBB_MSSM"
        result = preprocess_pipeline(path, "standard", False, True, True)
        processed, stats = result
        self.assertIsInstance(processed, pd.DataFrame)
        self.assertEqual(processed.shape[1], 13)
        self.assertFalse(processed.isnull().values.any())

    def test_mri_preprocessing(self):
        preprocess_pipeline()
        path = "data/processed/mri.pt"
        self.assertTrue(os.path.exists(path))
        data = torch.load(path)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertTrue(all(t[1].shape[1:] == torch.Size([128, 128]) for t in data))



    
    