import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for the GMM
alphas = [0.5, 0.3, 0.2]
means = [-3, 0, 2]
stds = [2, 1, 4]
num_samples = 10000
burn_in = 50  # Drop the first 50 datapoints

# Define the Gaussian Mixture Model PDF
def gmm_pdf(x):
    return sum(alpha * norm.pdf(x, mean, std) for alpha, mean, std in zip(alphas, means, stds))

# Metropolis algorithm
def metropolis_sampling(pdf, num_samples, proposal_std=1.0):
    samples = []
    current_sample = np.random.uniform(-10, 10)  # Initial sample
    for _ in range(num_samples + burn_in):  # Include burn-in samples
        proposed_sample = np.random.normal(current_sample, proposal_std)
        acceptance_ratio = pdf(proposed_sample) / pdf(current_sample)
        if np.random.rand() < acceptance_ratio:
            current_sample = proposed_sample
        samples.append(current_sample)
    return np.array(samples[burn_in:])  # Remove burn-in samples

# Generate samples using Metropolis algorithm
samples = metropolis_sampling(gmm_pdf, num_samples)

# Plot the histogram of samples and the GMM PDF
x = np.linspace(-10, 10, 1000)
gmm_values = gmm_pdf(x)

plt.figure(figsize=(12, 6))
plt.hist(samples, bins=50, density=True, alpha=0.6, label='Histogram of Samples')
plt.plot(x, gmm_values, label='Gaussian Mixture Model PDF', color='red', linewidth=2)
plt.title("Metropolis Sampling vs GMM PDF")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.show()
