# imhk_sampler.py
# IMHK sampler (Independent Metropolis-Hastings-Klein) for lattice Gaussian over Z^2

import numpy as np
from numpy.linalg import qr, norm
from scipy.stats import norm as gaussian
import matplotlib.pyplot as plt

# -------------------------------
# Discrete Gaussian (unnormalized)
# -------------------------------
def discrete_gaussian_pdf(x, c, sigma):
    return np.exp(-np.sum((x - c)**2) / (2 * sigma**2))

# -------------------------------
# Klein's Algorithm (Proposal)
# -------------------------------
def kleins_algorithm(B, sigma, c):
    n = B.shape[1]
    Q, R = qr(B)
    y = np.zeros(n)
    c_ = np.dot(np.linalg.inv(R), np.dot(Q.T, c))

    for i in reversed(range(n)):
        sigma_i = sigma / abs(R[i, i])
        z = gaussian.rvs(loc=c_[i], scale=sigma_i)
        y[i] = np.round(z)

    return np.dot(B, y)

# -------------------------------
# IMHK Sampler
# -------------------------------
def imhk_sampler(B, sigma, c, num_samples):
    samples = []
    rejections = 0
    current = np.zeros(B.shape[1])

    for _ in range(num_samples):
        proposal = kleins_algorithm(B, sigma, c)
        alpha = min(1.0,
                    discrete_gaussian_pdf(proposal, c, sigma) /
                    discrete_gaussian_pdf(current, c, sigma))

        if np.random.rand() < alpha:
            current = proposal
        else:
            rejections += 1

        samples.append(current.copy())

    return np.array(samples), rejections / num_samples

# -------------------------------
# Run + Save + Plot
# -------------------------------
if __name__ == "__main__":
    B = np.array([[1, 0], [0, 1]])  # standard Z^2 lattice
    sigma = 2.0
    c = np.array([0.0, 0.0])
    num_samples = 1000

    samples, rejection_rate = imhk_sampler(B, sigma, c, num_samples)

    np.save("results/logs/samples_imhk.npy", samples)

    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
    plt.title(f"IMHK Sampling (σ={sigma})\nRejection rate: {rejection_rate:.2f}")
    plt.grid(True)
    plt.axis("equal")
    plt.savefig("results/plots/imhk_scatter.png")
    plt.show()
