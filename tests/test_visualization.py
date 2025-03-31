import unittest
import torch
import numpy as np
import os
from adVAE.visualization.plot_utils import plot_loss_curve, plot_accuracy_curve, plot_reconstruction_distribution


class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.out_dir = "results/tests"
        os.makedirs(self.out_dir, exist_ok=True)
        self.accuracy = [0.1, 0.2, 0.3]
        self.losses = [1.0, 0.8, 0.6]
        self.original = np.random.rand(100, 13)
        self.reconstructed = self.original + np.random.normal(0, 0.1, self.original.shape)

    def test_plot_loss_curve(self):
        path = os.path.join(self.out_dir, "loss_curve.png")
        plot_loss_curve(self.losses, save_path=path)
        self.assertTrue(os.path.exists(path))

    def test_plot_accuracy_curve(self):
        path = os.path.join(self.out_dir, "acc_curve.png")
        plot_accuracy_curve(self.accuracy, save_path=path)
        self.assertTrue(os.path.exists(path))

    def test_plot_reconstruction_distribution(self):
        path = os.path.join(self.out_dir, "recon_dist.png")
        plot_reconstruction_distribution(self.original, self.reconstructed, save_path=path, title_prefix="Test")
        self.assertTrue(os.path.exists(path))
