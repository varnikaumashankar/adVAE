import unittest
from adVAE.metrics.evaluation_metrics import reconstruction_error
import numpy as np

class TestMetrics(unittest.TestCase):
    def test_reconstruction_error(self):
        real = np.random.rand(5, 13)
        generated = np.random.rand(5, 13)
        mse = reconstruction_error(real, generated)
        self.assertTrue(mse >= 0)