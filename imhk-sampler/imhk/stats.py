from sage.structure.element import Vector
import numpy as np
from sage.all import *

def compute_total_variation_distance(samples, sigma, lattice_basis, center=None):
    """
    Compute an approximation of the total variation distance between the empirical
    distribution of the samples and the ideal discrete Gaussian.
    
    Args:
        samples: List of samples
        sigma: The standard deviation used for the Gaussian
        lattice_basis: The lattice basis
        center: Center of the Gaussian distribution
        
    Returns:
        The estimated total variation distance
    """
    from .utils import discrete_gaussian_pdf

    n_dims = len(samples[0])
    if center is None:
        center = vector(RR, [0] * n_dims)
    
    # Compute the empirical distribution
    sample_counts = {}
    for sample in samples:
        sample_tuple = tuple(map(float, sample))
        sample_counts[sample_tuple] = sample_counts.get(sample_tuple, 0) + 1
    
    # Normalize to get empirical probabilities
    total_samples = len(samples)
    empirical_probs = {k: v / total_samples for k, v in sample_counts.items()}
    
    # Compute the theoretical probabilities for each observed point
    Z = 0  # Normalization constant
    theoretical_probs = {}
    
    # First pass: compute normalization constant Z
    unique_points = set(empirical_probs.keys())
    for point in unique_points:
        point_vector = vector(RR, point)
        density = discrete_gaussian_pdf(point_vector, sigma, center)
        Z += density
    
    # Second pass: compute normalized probabilities
    for point in unique_points:
        point_vector = vector(RR, point)
        density = discrete_gaussian_pdf(point_vector, sigma, center)
        theoretical_probs[point] = density / Z
    
    # Compute the total variation distance
    tv_distance = 0.5 * sum(abs(empirical_probs.get(point, 0) - theoretical_probs.get(point, 0))
                           for point in unique_points)
    
    return tv_distance

def compute_kl_divergence(samples, sigma, lattice_basis, center=None):
    """
    Compute an approximation of the KL divergence between the empirical
    distribution of the samples and the ideal discrete Gaussian.
    
    Args:
        samples: List of samples
        sigma: The standard deviation used for the Gaussian
        lattice_basis: The lattice basis
        center: Center of the Gaussian distribution
        
    Returns:
        The estimated KL divergence
    """
    from .utils import discrete_gaussian_pdf

    n_dims = len(samples[0])
    if center is None:
        center = vector(RR, [0] * n_dims)
    
    # Compute the empirical distribution
    sample_counts = {}
    for sample in samples:
        sample_tuple = tuple(map(float, sample))
        sample_counts[sample_tuple] = sample_counts.get(sample_tuple, 0) + 1
    
    # Normalize to get empirical probabilities
    total_samples = len(samples)
    empirical_probs = {k: v / total_samples for k, v in sample_counts.items()}
    
    # Compute the theoretical probabilities for each observed point
    Z = 0  # Normalization constant
    theoretical_probs = {}
    
    # First pass: compute normalization constant
    unique_points = set(empirical_probs.keys())
    for point in unique_points:
        point_vector = vector(RR, point)
        density = discrete_gaussian_pdf(point_vector, sigma, center)
        Z += density
    
    # Second pass: compute normalized probabilities
    for point in unique_points:
        point_vector = vector(RR, point)
        density = discrete_gaussian_pdf(point_vector, sigma, center)
        theoretical_probs[point] = density / Z
    
    # Compute the KL divergence
    kl_divergence = sum(p * log(p / theoretical_probs.get(point, 1e-10))
                        for point, p in empirical_probs.items()
                        if theoretical_probs.get(point, 0) > 0)
    
    return kl_divergence