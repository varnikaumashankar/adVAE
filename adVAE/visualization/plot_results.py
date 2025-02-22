import matplotlib.pyplot as plt

def plot_reconstruction(original, reconstructed, num_samples=10):
    """
    Plots the original and reconstructed samples side by side for comparison.

    Args:
        original: The original data samples.
        reconstructed: The reconstructed data samples.
        num_samples (int): The number of samples to plot. Default is 10.
        
    Returns:
        None
    """

    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.plot(original[i])
        plt.title("Original")
        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.plot(reconstructed[i])
        plt.title("Reconstructed")
    plt.tight_layout()
    plt.show()

