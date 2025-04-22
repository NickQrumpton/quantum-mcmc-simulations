#!/usr/bin/env sage
# -*- coding: utf-8 -*-

"""
Independent Metropolis-Hastings-Klein (IMHK) Sampler for Discrete Gaussian Distributions over Lattices

This module implements the IMHK algorithm for sampling from discrete Gaussian distributions
over lattices, along with various diagnostics, visualization tools, and comparisons with Klein's algorithm.
The implementation is primarily for research in small dimensions (2-4D) to establish empirical baselines
against which quantum speedups can be measured.

References:
[1] Klein, P. (2000). Finding the closest lattice vector when it's unusually close.
    In Proceedings of SODA'00, 937-941.
[2] Aggarwal, D., Dadush, D., Regev, O., & Stephens-Davidowitz, N. (2015).
    Solving the shortest vector problem in 2^n time using discrete Gaussian sampling.
    In Proceedings of STOC'15, 733-742.
[3] Regev, O. (2009). On lattices, learning with errors, random linear codes, and cryptography.
    Journal of the ACM, 56(6), 1-40.

The implementation includes:
- IMHK and Klein samplers
- Convergence diagnostics (trace plots, autocorrelation, ESS, Gelman-Rubin, Geweke)
- Statistical distance measures (Total Variation, KL divergence)
- Visualization tools
- Parameter sweep experiments
- Theoretical bound comparisons
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import os
import time
import pickle
import multiprocessing
from functools import partial
import cProfile
import pstats
from io import StringIO
from scipy.linalg import eigh
from sage.all import *

# Create directories for results
os.makedirs('results/logs', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/profiles', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
set_random_seed(42)

# ----------------------------------------
# Core IMHK and Klein Sampler Implementation
# ----------------------------------------

def discrete_gaussian_pdf(x, sigma, center=None):
    """
    Compute the probability density function of a discrete Gaussian distribution.
    
    The discrete Gaussian distribution over Z^n with center c and parameter σ has density
    proportional to exp(-||x-c||^2/2σ^2) for each x ∈ Z^n. This is a fundamental
    distribution in lattice-based cryptography and the basis for many sampling algorithms.
    
    See Section 2.1 of [2] for formal definitions.
    
    Args:
        x: The point to evaluate (can be a vector or scalar)
        sigma: The standard deviation of the Gaussian
        center: The center of the Gaussian (default: origin)
        
    Returns:
        The probability density at point x
    """
    if center is None:
        if isinstance(x, (list, tuple, np.ndarray)) or isinstance(x, sage.modules.vector_real_dense.Vector_real_dense):
            center = vector(RR, [0] * len(x))
        else:
            center = 0
    
    if isinstance(x, (list, tuple, np.ndarray)):
        x = vector(RR, x)
    
    if isinstance(x, sage.modules.vector_real_dense.Vector_real_dense):
        # Compute the squared norm of (x - center)
        squared_norm = sum((xi - ci) ** 2 for xi, ci in zip(x, center))
        return exp(-squared_norm / (2 * sigma ** 2))
    else:
        # Scalar case
        return exp(-(x - center) ** 2 / (2 * sigma ** 2))

def precompute_discrete_gaussian_probabilities(sigma, center=0, radius=6):
    """
    Precompute discrete Gaussian probabilities for integers within radius*sigma of center.
    
    This optimization is crucial for higher-dimensional sampling where we need
    repeated evaluations of the discrete Gaussian PDF. By precomputing probabilities
    for a reasonable range, we can significantly reduce computation time.
    
    Args:
        sigma: The standard deviation of the Gaussian
        center: The center of the Gaussian (default: 0)
        radius: How many standard deviations to consider (default: 6)
        
    Returns:
        A dictionary mapping integer points to their probabilities
    """
    lower_bound = int(floor(center - radius * sigma))
    upper_bound = int(ceil(center + radius * sigma))
    
    probs = {}
    for x in range(lower_bound, upper_bound + 1):
        probs[x] = discrete_gaussian_pdf(x, sigma, center)
    
    # Normalize
    total = sum(probs.values())
    for x in probs:
        probs[x] /= total
    
    return probs

def discrete_gaussian_sampler_1d(center, sigma):
    """
    Sample from a 1D discrete Gaussian distribution centered at 'center' with width 'sigma'.
    Uses an efficient table-based approach with rejection sampling for tails.
    
    This is a key building block for Klein's algorithm. For high dimensions (n > 10),
    consider implementing Karney's method or other advanced 1D sampling techniques.
    
    Args:
        center: The center of the Gaussian
        sigma: The standard deviation
        
    Returns:
        An integer sampled from the discrete Gaussian
    """
    # Compute the integer center
    c_int = int(round(center))
    center_frac = center - c_int
    
    # For very small sigma, just return the rounded center
    if sigma < 0.01:
        return c_int
    
    # For standard cases, use precomputed probabilities within 6*sigma
    # This truncation is justified as the probability mass beyond 6σ is negligible
    # (less than 2e-9 for each tail)
    tau = 6
    lower_bound = int(floor(center - tau * sigma))
    upper_bound = int(ceil(center + tau * sigma))
    
    # Precompute probabilities for this range
    probabilities = [(x, discrete_gaussian_pdf(x, sigma, center)) 
                     for x in range(lower_bound, upper_bound + 1)]
    
    # Normalize probabilities
    total = sum(p for _, p in probabilities)
    
    # Sample using inverse transform sampling
    u = random()
    cum_prob = 0
    for x, prob in probabilities:
        cum_prob += prob / total
        if u <= cum_prob:
            return x
    
    # Fallback (should rarely happen)
    return c_int

def klein_sampler(B, sigma, c=None):
    """
    Klein's algorithm for sampling from a discrete Gaussian over a lattice.
    
    Klein's algorithm [1] samples approximately from a discrete Gaussian over a lattice
    by recursively sampling each coordinate from a 1D discrete Gaussian, conditional
    on previously sampled coordinates. While not exact, it provides samples with density
    approximately proportional to the discrete Gaussian when σ is sufficiently large
    relative to the orthogonalized lattice vectors.
    
    For high dimensions (n > 30), consider implementing more advanced techniques like:
    1. Peikert's convolution sampler
    2. GenSample algorithm
    3. The Gibbs variant in Section 4 of [2]
    
    Args:
        B: The lattice basis matrix (rows are basis vectors)
        sigma: The standard deviation of the Gaussian
        c: The center of the Gaussian (default: origin)
        
    Returns:
        A lattice point sampled according to Klein's algorithm
    """
    n = B.nrows()
    if c is None:
        c = vector(RR, [0] * n)
    else:
        c = vector(RR, c)
    
    # Convert to a ring that supports Gram-Schmidt orthogonalization
    # SageMath requires exact rings like QQ for gram_schmidt()
    B_copy = matrix(QQ, B)  # Use rational numbers for exact computation
    
    # Gram-Schmidt orthogonalization
    GSO = B_copy.gram_schmidt()
    Q = GSO[0]  # Orthogonal basis
    
    # Sample from discrete Gaussian over Z^n, working backwards
    z = vector(ZZ, [0] * n)
    c_prime = vector(RR, c)
    
    for i in range(n-1, -1, -1):
        b_i_star = vector(RR, Q.row(i))  # Convert to RR for numerical stability
        b_i = vector(RR, B_copy.row(i))
        
        # Project c_prime onto b_i_star
        # μ = <c', b*>/||b*||^2
        b_star_norm_sq = b_i_star.dot_product(b_i_star)
        mu_i = b_i_star.dot_product(c_prime) / b_star_norm_sq
        
        # Set sigma_i for this coordinate (σ / ||b*||)
        sigma_i = sigma / sqrt(b_star_norm_sq)
        
        # Sample from 1D discrete Gaussian
        z_i = discrete_gaussian_sampler_1d(mu_i, sigma_i)
        z[i] = z_i
        
        # Update c_prime for the next iteration
        c_prime = c_prime - z_i * b_i
    
    # Convert z to a lattice point by multiplying with the basis
    return B * z

def imhk_sampler(B, sigma, num_samples, c=None, burn_in=1000, num_chains=1, initial_samples=None):
    """
    Independent Metropolis-Hastings-Klein algorithm for sampling from a discrete Gaussian
    over a lattice.
    
    This algorithm uses Klein's algorithm as the proposal distribution in a Metropolis-Hastings
    framework. This produces samples exactly from the discrete Gaussian (asymptotically in the
    limit of infinite samples), correcting for any inaccuracy in Klein's algorithm.
    
    The choice of burn-in and number of samples should be guided by:
    1. The Gelman-Rubin statistic approaching 1.0
    2. The mixing time estimate from autocorrelation or TV distance
    3. Effective Sample Size (ESS) requirements for the intended analysis
    
    For dimensions n > 30, consider:
    - Parallelizing multiple chains using multiprocessing
    - Using sparse matrix representations for the basis
    - Implementing adaptive proposal distributions
    
    Args:
        B: The lattice basis matrix (rows are basis vectors)
        sigma: The standard deviation of the Gaussian
        num_samples: The number of samples to generate
        c: The center of the Gaussian (default: origin)
        burn_in: The number of initial samples to discard
        num_chains: Number of independent chains to run (for diagnostics)
        initial_samples: Optional list of initial samples for each chain
        
    Returns:
        A list of lattice points sampled according to the discrete Gaussian,
        along with acceptance rate and other diagnostics
    """
    n = B.nrows()
    if c is None:
        c = vector(RR, [0] * n)
    else:
        c = vector(RR, c)
    
    # Set up chains
    all_chains = []
    all_accepts = []
    all_densities = []
    
    for chain_idx in range(num_chains):
        # Initialize the chain with a sample from Klein's algorithm or provided initial sample
        if initial_samples is not None and chain_idx < len(initial_samples):
            current_sample = initial_samples[chain_idx]
        else:
            current_sample = klein_sampler(B, sigma, c)
        
        current_density = discrete_gaussian_pdf(current_sample, sigma, c)
        
        samples = []
        acceptance_count = 0
        total_count = 0
        
        # Monitor individual samples for diagnostics
        chain_samples = []  # Store all samples including burn-in for diagnostics
        chain_accepts = []  # Store whether each proposal was accepted
        chain_densities = []  # Store the density of each sample
        
        # Run the chain
        for i in range(num_samples + burn_in):
            # Generate proposal using Klein's algorithm
            proposal = klein_sampler(B, sigma, c)
            proposal_density = discrete_gaussian_pdf(proposal, sigma, c)
            
            # Compute the Metropolis-Hastings acceptance ratio
            # For independent proposals with target π and proposal q:
            # acceptance probability = min(1, π(y)q(x)/π(x)q(y))
            # In our case with Klein's algorithm, we approximate q(x) ∝ exp(-||x-c||²/2σ²), 
            # which is close to the target π but not exactly the same due to discretization
            # and the non-optimal nature of Klein's algorithm
            ratio = proposal_density / current_density if current_density > 0 else 1.0
            
            # Accept or reject the proposal
            accept = random() < min(1, ratio)
            if accept:
                current_sample = proposal
                current_density = proposal_density
                acceptance_count += 1
            
            total_count += 1
            chain_accepts.append(accept)
            chain_samples.append(current_sample)
            chain_densities.append(current_density)
            
            # Store the sample if we're past the burn-in period
            if i >= burn_in:
                samples.append(current_sample)
        
        all_chains.append(samples)
        all_accepts.append(chain_accepts)
        all_densities.append(chain_densities)
    
    # If only one chain, return in the original format
    if num_chains == 1:
        acceptance_rate = sum(all_accepts[0]) / len(all_accepts[0])
        return all_chains[0], acceptance_rate, chain_samples, chain_accepts
    
    # Otherwise, return all chains and diagnostics
    acceptance_rates = [sum(accepts) / len(accepts) for accepts in all_accepts]
    
    return all_chains, acceptance_rates, all_accepts, all_densities

def estimate_mixing_time(B, sigma, max_iterations=10000, threshold=0.1, c=None):
    """
    Estimate the mixing time of the IMHK algorithm by tracking the total variation
    distance to the target distribution over iterations.
    
    The mixing time is defined as the number of iterations needed for the TV distance
    to drop below a threshold (typically 0.1 or 0.01). This function runs a long chain
    and tracks the empirical TV distance at regular intervals.
    
    Args:
        B: The lattice basis matrix (rows are basis vectors)
        sigma: The standard deviation of the Gaussian
        max_iterations: Maximum number of iterations to run
        threshold: TV distance threshold for determining convergence
        c: The center of the Gaussian (default: origin)
        
    Returns:
        Estimated mixing time and a list of (iteration, TV distance) pairs
    """
    n = B.nrows()
    if c is None:
        c = vector(RR, [0] * n)
    else:
        c = vector(RR, c)
    
    # Generate a large number of reference samples from Klein's algorithm
    # These serve as an approximation to the target distribution
    num_reference = 10000
    reference_samples = [klein_sampler(B, sigma, c) for _ in range(num_reference)]
    
    # Initialize the chain with a sample from Klein's algorithm
    current_sample = klein_sampler(B, sigma, c)
    current_density = discrete_gaussian_pdf(current_sample, sigma, c)
    
    collected_samples = []
    tv_distances = []
    mixing_time = max_iterations
    
    # Run the chain and track TV distance
    check_interval = min(100, max_iterations // 100)  # Check TV distance every 100 iterations
    
    for i in range(max_iterations):
        # Generate proposal using Klein's algorithm
        proposal = klein_sampler(B, sigma, c)
        proposal_density = discrete_gaussian_pdf(proposal, sigma, c)
        
        # Compute acceptance ratio
        ratio = proposal_density / current_density if current_density > 0 else 1.0
        
        # Accept or reject
        accept = random() < min(1, ratio)
        if accept:
            current_sample = proposal
            current_density = proposal_density
        
        # Collect sample
        collected_samples.append(current_sample)
        
        # Check TV distance periodically
        if (i+1) % check_interval == 0 or i == max_iterations - 1:
            # Use the samples collected so far to estimate TV distance
            tv_dist = compute_empirical_tv_distance(collected_samples, reference_samples)
            tv_distances.append((i+1, tv_dist))
            
            # Check if we've reached the threshold
            if tv_dist < threshold and mixing_time == max_iterations:
                mixing_time = i+1
    
    return mixing_time, tv_distances

def compute_empirical_tv_distance(samples1, samples2):
    """
    Compute the total variation distance between two empirical distributions.
    
    Args:
        samples1: First set of samples
        samples2: Second set of samples
        
    Returns:
        The estimated total variation distance
    """
    # Count occurrences of each point in both sample sets
    counts1 = {}
    for sample in samples1:
        sample_tuple = tuple(map(float, sample))
        counts1[sample_tuple] = counts1.get(sample_tuple, 0) + 1
    
    counts2 = {}
    for sample in samples2:
        sample_tuple = tuple(map(float, sample))
        counts2[sample_tuple] = counts2.get(sample_tuple, 0) + 1
    
    # Get all unique points
    all_points = set(counts1.keys()) | set(counts2.keys())
    
    # Convert to probabilities
    total1 = len(samples1)
    total2 = len(samples2)
    
    probs1 = {point: counts1.get(point, 0) / total1 for point in all_points}
    probs2 = {point: counts2.get(point, 0) / total2 for point in all_points}
    
    # Compute TV distance
    tv_distance = 0.5 * sum(abs(probs1[point] - probs2[point]) for point in all_points)
    
    return tv_distance

def compute_gelman_rubin(chains):
    """
    Compute the Gelman-Rubin statistic (R̂) for assessing MCMC convergence across multiple chains.
    
    The Gelman-Rubin statistic (also called the potential scale reduction factor) compares
    the within-chain variance to the between-chain variance. Values close to 1 indicate
    convergence. This diagnostic is crucial for ensuring reliable results in MCMC sampling.
    
    Args:
        chains: List of chains, where each chain is a list of samples
        
    Returns:
        A list of R-hat values, one for each dimension
    """
    if len(chains) < 2:
        raise ValueError("At least 2 chains are required for Gelman-Rubin diagnostic")
    
    n_chains = len(chains)
    chain_length = min(len(chain) for chain in chains)
    n_dims = len(chains[0][0])
    
    # Convert chains to numpy arrays for easier computation
    chains_np = np.zeros((n_chains, chain_length, n_dims))
    for i, chain in enumerate(chains):
        for j in range(chain_length):
            chains_np[i, j] = np.array([float(x) for x in chain[j]])
    
    # Compute R-hat for each dimension
    r_hat = np.zeros(n_dims)
    
    for dim in range(n_dims):
        # Extract this dimension's values from all chains
        dim_chains = chains_np[:, :, dim]
        
        # Compute mean of each chain
        chain_means = np.mean(dim_chains, axis=1)
        
        # Compute overall mean
        overall_mean = np.mean(chain_means)
        
        # Compute between-chain variance (B)
        B = chain_length * np.sum((chain_means - overall_mean)**2) / (n_chains - 1)
        
        # Compute within-chain variance (W)
        chain_vars = np.zeros(n_chains)
        for i in range(n_chains):
            chain_vars[i] = np.sum((dim_chains[i] - chain_means[i])**2) / (chain_length - 1)
        W = np.mean(chain_vars)
        
        # Compute pooled variance estimate
        V = ((chain_length - 1) / chain_length) * W + B / chain_length
        
        # Compute R-hat
        r_hat[dim] = np.sqrt(V / W)
    
    return r_hat

def compute_geweke(samples, first=0.1, last=0.5):
    """
    Compute the Geweke diagnostic for assessing stationarity within a single chain.
    
    The Geweke diagnostic compares the means of the first and last parts of a Markov chain
    to check for stationarity. A value outside of [-1.96, 1.96] suggests that the chain
    has not converged to the stationary distribution.
    
    Args:
        samples: List of samples from a Markov chain
        first: Fraction of the chain to use for the first segment (default: 0.1)
        last: Fraction of the chain to use for the last segment (default: 0.5)
        
    Returns:
        A list of Geweke z-scores, one for each dimension
    """
    n_samples = len(samples)
    n_dims = len(samples[0])
    
    # Convert samples to numpy array
    samples_np = np.zeros((n_samples, n_dims))
    for i, sample in enumerate(samples):
        samples_np[i] = np.array([float(x) for x in sample])
    
    # Determine segment indices
    n_first = int(first * n_samples)
    n_last = int(last * n_samples)
    first_samples = samples_np[:n_first]
    last_samples = samples_np[n_samples-n_last:]
    
    # Compute Geweke z-scores for each dimension
    z_scores = np.zeros(n_dims)
    
    for dim in range(n_dims):
        # Extract this dimension's values
        first_vals = first_samples[:, dim]
        last_vals = last_samples[:, dim]
        
        # Compute means
        first_mean = np.mean(first_vals)
        last_mean = np.mean(last_vals)
        
        # Compute spectral density estimates at frequency zero
        # Using simple standard error of the mean as an approximation
        first_se = np.std(first_vals) / np.sqrt(n_first)
        last_se = np.std(last_vals) / np.sqrt(n_last)
        
        # Compute z-score
        z_scores[dim] = (first_mean - last_mean) / np.sqrt(first_se**2 + last_se**2)
    
    return z_scores

def compute_spectral_gap(B, sigma, n_samples=1000, max_states=1000, c=None):
    """
    Estimate the spectral gap of the IMHK transition matrix for small lattices.
    
    The spectral gap determines the convergence rate of the Markov chain. A larger
    gap implies faster mixing. For small lattices, we can approximate this by building
    an empirical transition matrix and finding its second-largest eigenvalue.
    
    Args:
        B: The lattice basis matrix
        sigma: The standard deviation of the Gaussian
        n_samples: Number of samples to use for transition matrix estimation
        max_states: Maximum number of states to consider
        c: The center of the Gaussian (default: origin)
        
    Returns:
        Estimated spectral gap and the transition matrix
    """
    if B.nrows() > 3:
        raise ValueError("Computing spectral gap is only feasible for small dimensions (≤ 3)")
    
    # Generate samples to determine the state space
    samples, _, _, _ = imhk_sampler(B, sigma, n_samples, c, burn_in=n_samples//10)
    
    # Identify the most common points (states)
    point_counts = {}
    for sample in samples:
        sample_tuple = tuple(map(float, sample))
        point_counts[sample_tuple] = point_counts.get(sample_tuple, 0) + 1
    
    # Sort by frequency and take the top max_states
    most_common = sorted(point_counts.items(), key=lambda x: x[1], reverse=True)[:max_states]
    states = [state for state, _ in most_common]
    
    # Create a mapping from states to indices
    state_to_idx = {state: i for i, state in enumerate(states)}
    
    # Estimate transition probabilities
    n_states = len(states)
    transitions = np.zeros((n_states, n_states))
    
    # For each state, estimate transitions using Klein's proposal and MH acceptance
    for i, state in enumerate(states):
        state_vector = vector(RR, state)
        state_density = discrete_gaussian_pdf(state_vector, sigma, c)
        
        # Generate a large number of proposals and compute acceptance probabilities
        n_proposals = 1000
        for _ in range(n_proposals):
            proposal = klein_sampler(B, sigma, c)
            proposal_tuple = tuple(map(float, proposal))
            
            if proposal_tuple in state_to_idx:
                j = state_to_idx[proposal_tuple]
                proposal_density = discrete_gaussian_pdf(proposal, sigma, c)
                
                # Compute acceptance probability
                ratio = proposal_density / state_density if state_density > 0 else 1.0
                acceptance_prob = min(1, ratio)
                
                # Update transition matrix
                transitions[i, j] += acceptance_prob / n_proposals
        
        # Self-transition probability (rejection)
        transitions[i, i] += 1 - transitions[i].sum()
    
    # Ensure transition matrix is row-stochastic
    for i in range(n_states):
        transitions[i] = transitions[i] / transitions[i].sum()
    
    # Compute eigenvalues
    eigvals = np.linalg.eigvals(transitions)
    eigvals = sorted(np.abs(eigvals), reverse=True)
    
    # The spectral gap is 1 - λ_2, where λ_2 is the second-largest eigenvalue
    spectral_gap = 1 - eigvals[1]
    
    return spectral_gap, transitions

# ----------------------------------------
# Lattice Basis Generation Functions
# ----------------------------------------

def generate_random_lattice(dim, bit_size=10, det_range=(10, 100)):
    """
    Generate a random integer lattice basis with controlled determinant.
    
    Args:
        dim: The dimension of the lattice
        bit_size: The bit size of matrix entries
        det_range: Range for the determinant
        
    Returns:
        A random lattice basis matrix
    """
    # Generate a random matrix with integer entries
    B = random_matrix(ZZ, dim, dim, x=-2**bit_size, y=2**bit_size)
    
    # Ensure the determinant is in the desired range
    current_det = abs(B.determinant())
    target_det = randint(det_range[0], det_range[1])
    
    if current_det != 0:
        # Scale the first row to achieve the target determinant
        scale_factor = target_det / current_det
        B[0] = B[0] * scale_factor
    else:
        # If determinant is zero, create a new basis
        return generate_random_lattice(dim, bit_size, det_range)
    
    return B

def generate_ntru_lattice(dim, q=197):
    """
    Generate an NTRU-like lattice for cryptographic testing.
    
    NTRU lattices have the form:
    [I  H]
    [0 qI]
    where H is a circulant matrix.
    
    This is a simplified version for testing purposes.
    
    Args:
        dim: The dimension of the lattice (must be even)
        q: The modulus (prime number)
        
    Returns:
        An NTRU-like lattice basis
    """
    if dim % 2 != 0:
        raise ValueError("NTRU dimension must be even")
    
    n = dim // 2
    
    # Create an n×n circulant matrix H with small entries
    h = [randint(-3, 3) for _ in range(n)]
    H = matrix(ZZ, n, n)
    for i in range(n):
        for j in range(n):
            H[i, j] = h[(j - i) % n]
    
    # Create the NTRU basis
    B = matrix(ZZ, dim, dim)
    
    # Top-left: Identity
    for i in range(n):
        B[i, i] = 1
    
    # Top-right: H
    for i in range(n):
        for j in range(n):
            B[i, n + j] = H[i, j]
    
    # Bottom-right: q*Identity
    for i in range(n):
        B[n + i, n + i] = q
    
    return B

def apply_lattice_reduction(B, method='LLL', block_size=10):
    """
    Apply lattice basis reduction to improve sampling performance.
    
    Args:
        B: The lattice basis matrix
        method: Reduction method ('LLL' or 'BKZ')
        block_size: Block size for BKZ reduction
        
    Returns:
        The reduced lattice basis
    """
    B_reduced = copy(B)
    
    if method == 'LLL':
        B_reduced = B_reduced.LLL()
    elif method == 'BKZ':
        B_reduced = B_reduced.BKZ(block_size=block_size)
    else:
        raise ValueError(f"Unknown reduction method: {method}")
    
    return B_reduced

# ----------------------------------------
# Diagnostics Implementation
# ----------------------------------------

# ----------------------------------------
# Diagnostics Implementation
# ----------------------------------------

def compute_autocorrelation(samples, lag=50):
    """
    Compute the autocorrelation of a chain of samples up to a given lag.
    
    Autocorrelation measures the correlation between samples separated by different lags,
    providing insight into how long the chain needs to run to generate effectively
    independent samples. High autocorrelation indicates slow mixing.
    
    Args:
        samples: List of samples
        lag: Maximum lag to compute autocorrelation for
        
    Returns:
        A list of autocorrelation values for each dimension
    """
    n_dims = len(samples[0])
    acf_by_dim = []
    
    for dim in range(n_dims):
        # Extract this dimension's values
        dim_values = [float(sample[dim]) for sample in samples]
        dim_values = np.array(dim_values)
        
        # Compute autocorrelation
        acf = []
        mean = np.mean(dim_values)
        var = np.var(dim_values)
        
        for l in range(lag + 1):
            if l == 0:
                acf.append(1.0)
            else:
                # Compute autocorrelation at lag l
                sum_corr = 0
                for t in range(len(dim_values) - l):
                    sum_corr += (dim_values[t] - mean) * (dim_values[t + l] - mean)
                
                acf.append(sum_corr / ((len(dim_values) - l) * var))
        
        acf_by_dim.append(acf)
    
    return acf_by_dim

def compute_ess(samples):
    """
    Compute the Effective Sample Size (ESS) for each dimension of the samples.
    
    ESS estimates how many independent samples the Markov chain effectively provides.
    For a chain with autocorrelation, ESS will be lower than the actual number of samples.
    This is crucial for assessing estimation uncertainty from MCMC simulations.
    
    Args:
        samples: List of samples
        
    Returns:
        A list of ESS values, one for each dimension
    """
    n_dims = len(samples[0])
    ess_values = []
    
    for dim in range(n_dims):
        # Extract this dimension's values
        dim_values = [float(sample[dim]) for sample in samples]
        dim_values = np.array(dim_values)
        
        # Compute autocorrelation for lags 1 to 50 (or less if the series is shorter)
        max_lag = min(50, len(dim_values) // 4)
        acf = compute_autocorrelation([samples[i] for i in range(len(samples))], max_lag)[dim]
        
        # Find the lag where ACF cuts off (absolute value < 0.05 or becomes negative)
        cutoff_lag = max_lag
        for i in range(1, max_lag + 1):
            if abs(acf[i]) < 0.05 or acf[i] < 0:
                cutoff_lag = i
                break
        
        # Compute ESS using the truncated sum of autocorrelations
        rho_sum = 2 * sum(acf[1:cutoff_lag])
        ess = len(dim_values) / (1 + rho_sum)
        ess_values.append(ess)
    
    return ess_values

def plot_trace(samples, filename, title=None, tv_distances=None):
    """
    Create trace plots for each dimension of the samples, with optional TV distance plot.
    
    Trace plots show the sampled values over iterations and are a fundamental tool for
    assessing MCMC convergence visually. The addition of TV distance tracking allows
    quantitative assessment of convergence to the target distribution.
    
    Args:
        samples: List of samples
        filename: Filename to save the plot
        title: Optional title for the plot
        tv_distances: Optional list of (iteration, TV distance) pairs for convergence assessment
        
    Returns:
        None (saves plot to file)
    """
    n_dims = len(samples[0])
    
    # If we have TV distances, add an extra subplot for them
    n_plots = n_dims + (1 if tv_distances else 0)
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    for dim in range(n_dims):
        # Extract this dimension's values
        dim_values = [float(sample[dim]) for sample in samples]
        
        # Plot trace
        axes[dim].plot(dim_values)
        axes[dim].set_ylabel(f'Dimension {dim+1}')
        
        # Add horizontal lines for mean and quantiles
        mean_val = np.mean(dim_values)
        quantile_25 = np.percentile(dim_values, 25)
        quantile_75 = np.percentile(dim_values, 75)
        
        axes[dim].axhline(mean_val, color='r', linestyle='--', alpha=0.5, label='Mean')
        axes[dim].axhline(quantile_25, color='g', linestyle=':', alpha=0.3, label='25% Quantile')
        axes[dim].axhline(quantile_75, color='g', linestyle=':', alpha=0.3, label='75% Quantile')
        
        # Annotate with statistics
        axes[dim].text(0.02, 0.95, 
                       f'Mean: {mean_val:.2f}\nStd: {np.std(dim_values):.2f}\n'
                       f'Min: {min(dim_values):.2f}\nMax: {max(dim_values):.2f}', 
                       transform=axes[dim].transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if dim == 0:  # Only add legend to first plot to avoid repetition
            axes[dim].legend(loc='upper right')
    
    # If we have TV distances, add the TV distance plot
    if tv_distances:
        tv_ax = axes[-1]
        iterations, tv_values = zip(*tv_distances)
        tv_ax.plot(iterations, tv_values, 'b-', linewidth=2)
        tv_ax.set_ylabel('TV Distance')
        tv_ax.set_yscale('log')  # Log scale often better shows convergence rate
        tv_ax.grid(True, which='both', alpha=0.3)
        
        # Add a horizontal line at 0.1 for convergence threshold
        tv_ax.axhline(0.1, color='r', linestyle='--', alpha=0.7, label='Convergence Threshold')
        
        # Add a theoretical decay line for comparison (based on spectral gap)
        # Approximate as exp(-λ*t) where λ is the spectral gap
        # This is just an illustration - in practice, the spectral gap would be estimated
        if len(tv_values) > 2:
            # Simple estimate of decay rate from first few points
            decay_rate = -np.log(tv_values[min(10, len(tv_values)-1)] / tv_values[0]) / min(10, len(tv_values)-1)
            decay_rate = max(0.001, min(0.1, decay_rate))  # Constrain to reasonable values
            
            t = np.array(iterations)
            theory_line = tv_values[0] * np.exp(-decay_rate * t)
            tv_ax.plot(t, theory_line, 'r:', label=f'Exp. Decay (rate≈{decay_rate:.4f})')
            
            tv_ax.legend()
    
    axes[-1].set_xlabel('Iteration')
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.95)  # Adjust for the title
    plt.savefig(f'results/plots/{filename}')
    plt.close()

def plot_autocorrelation(acf_by_dim, filename, title=None):
    """
    Plot the autocorrelation function for each dimension.
    
    Autocorrelation plots are essential for understanding the mixing properties of the chain
    and determining the effective sample size. The decay rate of autocorrelation provides
    insights into the spectral properties of the transition matrix.
    
    Args:
        acf_by_dim: List of autocorrelation functions for each dimension
        filename: Filename to save the plot
        title: Optional title for the plot
        
    Returns:
        None (saves plot to file)
    """
    n_dims = len(acf_by_dim)
    
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]
    
    for dim in range(n_dims):
        axes[dim].plot(acf_by_dim[dim])
        axes[dim].set_ylabel(f'ACF Dim {dim+1}')
        
        # Add horizontal lines
        axes[dim].axhline(0, color='r', linestyle='--', alpha=0.3)
        axes[dim].axhline(0.05, color='g', linestyle=':', alpha=0.3)
        axes[dim].axhline(-0.05, color='g', linestyle=':', alpha=0.3)
        
        # Calculate effective sample size
        max_lag = len(acf_by_dim[dim]) - 1
        cutoff_lag = max_lag
        for i in range(1, max_lag + 1):
            if abs(acf_by_dim[dim][i]) < 0.05 or acf_by_dim[dim][i] < 0:
                cutoff_lag = i
                break
        
        # Add annotation with ESS calculation info
        corr_sum = 2 * sum(acf_by_dim[dim][1:cutoff_lag])
        ess = len(acf_by_dim[dim]) / (1 + corr_sum)
        
        axes[dim].text(0.7, 0.95, 
                       f'ESS: {ess:.1f}\nCutoff lag: {cutoff_lag}', 
                       transform=axes[dim].transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add exponential decay fit for comparison with theoretical mixing
        if max_lag > 5:
            # Fit an exponential decay to the autocorrelation
            x = np.arange(1, min(20, max_lag + 1))
            y = np.abs(acf_by_dim[dim][1:min(21, max_lag + 1)])
            
            # Avoid taking log of zero or negative values
            valid_idx = y > 0.05
            if np.sum(valid_idx) > 3:
                x_valid = x[valid_idx]
                y_valid = y[valid_idx]
                
                # Linear fit on log scale
                try:
                    log_y = np.log(y_valid)
                    coeffs = np.polyfit(x_valid, log_y, 1)
                    decay_rate = -coeffs[0]
                    
                    # Plot the fit
                    fit_x = np.arange(1, max_lag + 1)
                    fit_y = np.exp(coeffs[1]) * np.exp(-decay_rate * fit_x)
                    axes[dim].plot(fit_x, fit_y, 'r:', linewidth=1.5, 
                                   label=f'Exp. Decay (rate≈{decay_rate:.4f})')
                    
                    # Add legend to first dimension
                    if dim == 0:
                        axes[dim].legend(loc='upper right')
                except:
                    pass  # Skip if fit fails
    
    axes[-1].set_xlabel('Lag')
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.95)  # Adjust for the title
    plt.savefig(f'results/plots/{filename}')
    plt.close()

def plot_acceptance_trace(accepts, filename, window_size=100):
    """
    Plot the acceptance rate over time using a moving window, with trend analysis.
    
    The acceptance rate trajectory provides insights into the efficiency of the sampler
    and can help diagnose issues like poor proposal distributions or non-stationarity.
    Trend analysis helps identify if the chain is operating optimally.
    
    Args:
        accepts: List of booleans indicating acceptance
        filename: Filename to save the plot
        window_size: Size of the moving window
        
    Returns:
        None (saves plot to file)
    """
    accepts_arr = np.array(accepts, dtype=float)
    
    # Calculate moving average
    cumsum = np.cumsum(np.insert(accepts_arr, 0, 0))
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(moving_avg)) + window_size // 2
    ax.plot(x, moving_avg, label=f'Moving Avg (window={window_size})')
    
    # Add horizontal line for overall acceptance rate
    overall_rate = np.mean(accepts_arr)
    ax.axhline(overall_rate, color='r', linestyle='--', alpha=0.8, 
              label=f'Overall Rate: {overall_rate:.3f}')
    
    # Analyze trends in acceptance rate
    # Divide into three segments (early, middle, late)
    n_segments = 3
    segment_length = len(moving_avg) // n_segments
    
    segment_means = []
    for i in range(n_segments):
        start = i * segment_length
        end = (i + 1) * segment_length if i < n_segments - 1 else len(moving_avg)
        segment_mean = np.mean(moving_avg[start:end])
        segment_means.append(segment_mean)
    
    # Check for significant trend
    early_vs_late = (segment_means[-1] - segment_means[0]) / segment_means[0]
    
    # Add trend annotation
    if abs(early_vs_late) > 0.1:  # More than 10% change
        trend_direction = "increasing" if early_vs_late > 0 else "decreasing"
        ax.text(0.02, 0.02, 
               f"Trend: {trend_direction} ({early_vs_late:.1%} change)\n"
               f"Early: {segment_means[0]:.3f}, Late: {segment_means[-1]:.3f}",
               transform=ax.transAxes, 
               verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add diagnostic message
        if early_vs_late > 0.2:  # Strong increasing trend
            ax.text(0.98, 0.98, 
                   "Diagnostic: Chain might not have reached\n"
                   "stationary distribution yet. Consider\n"
                   "longer burn-in period.",
                   transform=ax.transAxes, 
                   horizontalalignment='right',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        elif early_vs_late < -0.2:  # Strong decreasing trend
            ax.text(0.98, 0.98, 
                   "Diagnostic: Acceptance rate declining.\n"
                   "This may indicate poor mixing or that the\n"
                   "proposal distribution is not optimal.",
                   transform=ax.transAxes, 
                   horizontalalignment='right',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        ax.text(0.02, 0.02, 
               f"Trend: stable (only {early_vs_late:.1%} change)\n"
               f"Early: {segment_means[0]:.3f}, Late: {segment_means[-1]:.3f}",
               transform=ax.transAxes, 
               verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Assess optimal acceptance rate
    # For MH with independent proposals, the optimal rate is higher than for random walk
    optimal_range = (0.2, 0.5)  # Independent MH typically works well in this range
    
    if overall_rate < optimal_range[0]:
        ax.text(0.98, 0.85, 
               f"Warning: Low acceptance rate ({overall_rate:.3f}).\n"
               f"Consider increasing σ or using a better proposal.",
               transform=ax.transAxes, 
               horizontalalignment='right',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    elif overall_rate > optimal_range[1]:
        if overall_rate > 0.7:
            message = "Very high acceptance rate might indicate\n" \
                      "redundant sampling or that σ is too small."
        else:
            message = "Acceptance rate is good for independent MH."
            
        ax.text(0.98, 0.85, 
               f"Note: {overall_rate:.3f} acceptance. {message}",
               transform=ax.transAxes, 
               horizontalalignment='right',
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('IMHK Acceptance Rate Over Time')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/{filename}')
    plt.close()

def plot_gelman_rubin(chains, filename, title=None):
    """
    Plot the Gelman-Rubin statistic for multiple chains.
    
    The Gelman-Rubin statistic (R̂) assesses convergence by comparing
    between-chain and within-chain variance. Values close to 1 indicate
    convergence.
    
    Args:
        chains: List of chains, where each chain is a list of samples
        filename: Filename to save the plot
        title: Optional title for the plot
        
    Returns:
        None (saves plot to file)
    """
    if len(chains) < 2:
        print("Warning: At least 2 chains are required for Gelman-Rubin plot")
        return
    
    n_dims = len(chains[0][0])
    
    # Compute R-hat for each dimension along the chains
    # We'll compute R-hat at different points to see convergence
    check_points = np.linspace(100, len(chains[0]), 20, dtype=int)
    r_hats = np.zeros((len(check_points), n_dims))
    
    for i, checkpoint in enumerate(check_points):
        # Use chains up to this checkpoint
        truncated_chains = [chain[:checkpoint] for chain in chains]
        r_hats[i] = compute_gelman_rubin(truncated_chains)
    
    # Plot R-hat for each dimension
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for dim in range(n_dims):
        ax.plot(check_points, r_hats[:, dim], 'o-', label=f'Dimension {dim+1}')
    
    # Add horizontal line at 1.1 (common threshold for convergence)
    ax.axhline(1.1, color='r', linestyle='--', alpha=0.5, label='Convergence Threshold (1.1)')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gelman-Rubin Statistic (R̂)')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Gelman-Rubin Convergence Diagnostic')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotation for final R̂ values
    final_r_hats = r_hats[-1]
    text = "Final R̂ values:\n"
    for dim in range(n_dims):
        text += f"Dim {dim+1}: {final_r_hats[dim]:.4f}"
        text += " ✓" if final_r_hats[dim] < 1.1 else " ✗"
        text += "\n"
    
    ax.text(0.02, 0.98, text,
           transform=ax.transAxes, 
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'results/plots/{filename}')
    plt.close()

def plot_geweke(samples, filename, title=None):
    """
    Plot the Geweke diagnostic for assessing stationarity within a single chain.
    
    The Geweke diagnostic compares means of early and late parts of the chain.
    Z-scores outside [-1.96, 1.96] suggest non-stationarity.
    
    Args:
        samples: List of samples from a Markov chain
        filename: Filename to save the plot
        title: Optional title for the plot
        
    Returns:
        None (saves plot to file)
    """
    n_dims = len(samples[0])
    
    # Compute Geweke z-scores for different segments
    window_fraction = 0.1  # Fraction of the chain for each window
    segments = np.linspace(0.5, 0.9, 5)  # End points for the last segment
    
    z_scores = np.zeros((len(segments), n_dims))
    
    for i, last_segment in enumerate(segments):
        z_scores[i] = compute_geweke(samples, first=window_fraction, last=last_segment)
    
    # Plot Geweke z-scores
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for dim in range(n_dims):
        ax.plot(segments, z_scores[:, dim], 'o-', label=f'Dimension {dim+1}')
    
    # Add horizontal lines at +/- 1.96 (95% confidence interval)
    ax.axhline(1.96, color='r', linestyle='--', alpha=0.5)
    ax.axhline(-1.96, color='r', linestyle='--', alpha=0.5, label='95% CI')
    
    ax.set_xlabel('End Point of Last Segment')
    ax.set_ylabel('Geweke Z-Score')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Geweke Convergence Diagnostic')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add conclusion text
    final_z_scores = z_scores[-1]
    converged = all(abs(z) < 1.96 for z in final_z_scores)
    
    if converged:
        conclusion = "✓ All dimensions appear stationary"
        color = 'lightgreen'
    else:
        conclusion = "✗ Some dimensions may not be stationary"
        color = 'lightcoral'
    
    ax.text(0.02, 0.02, conclusion,
           transform=ax.transAxes, 
           verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'results/plots/{filename}')
    plt.close()

# ----------------------------------------
# Visualization Implementation
# ----------------------------------------

def plot_2d_samples(samples, sigma, filename, lattice_basis=None, title=None, center=None, theoretical_dist=None):
    """
    Create a 2D scatter plot of the samples with optional density contours.
    
    This visualization provides a direct view of the sample distribution in 2D space,
    allowing visual assessment of how well the samples match the target distribution.
    For lattice-based cryptography, the smoothness of the sampling is particularly important.
    
    Args:
        samples: List of 2D samples
        sigma: The standard deviation used for the Gaussian
        filename: Filename to save the plot
        lattice_basis: The lattice basis (for plotting the fundamental domain)
        title: Optional title for the plot
        center: Center of the Gaussian distribution
        theoretical_dist: Optional dictionary mapping points to theoretical probabilities
        
    Returns:
        None (saves plot to file)
    """
    if center is None:
        center = [0, 0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract x and y coordinates
    x_coords = [float(sample[0]) for sample in samples]
    y_coords = [float(sample[1]) for sample in samples]
    
    # Calculate point density for color mapping
    from scipy.stats import gaussian_kde
    try:
        xy = np.vstack([x_coords, y_coords])
        z = gaussian_kde(xy)(xy)
        
        # Create a scatter plot with density-based coloring
        scatter = ax.scatter(x_coords, y_coords, c=z, alpha=0.7, 
                           cmap='viridis', s=30, edgecolor='k', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Point Density')
    except:
        # Fallback if KDE fails
        ax.scatter(x_coords, y_coords, alpha=0.7, s=30, edgecolor='k', linewidth=0.5)
    
    # Add density contours
    x_min, x_max = min(x_coords) - 2, max(x_coords) + 2
    y_min, y_max = min(y_coords) - 2, max(y_coords) + 2
    
    # Create a grid of points
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Compute the density at each grid point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = discrete_gaussian_pdf([X[i, j], Y[i, j]], sigma, center)
    
    # Plot contours
    contour = ax.contour(X, Y, Z, levels=10, cmap='coolwarm', alpha=0.5)
    
    # Plot the fundamental domain if basis is provided
    if lattice_basis is not None:
        origin = np.array([0, 0])
        v1 = np.array([float(lattice_basis[0, 0]), float(lattice_basis[0, 1])])
        v2 = np.array([float(lattice_basis[1, 0]), float(lattice_basis[1, 1])])
        
        # Plot basis vectors
        ax.arrow(origin[0], origin[1], v1[0], v1[1], head_width=0.2, head_length=0.3, 
               fc='blue', ec='blue', label='Basis Vector 1')
        ax.arrow(origin[0], origin[1], v2[0], v2[1], head_width=0.2, head_length=0.3, 
               fc='green', ec='green', label='Basis Vector 2')
        
        # Plot the parallelogram
        ax.plot([0, v1[0], v1[0]+v2[0], v2[0], 0], 
               [0, v1[1], v1[1]+v2[1], v2[1], 0], 
               'r--', alpha=0.7, label='Fundamental Domain')
    
    # Mark the center
    ax.scatter([center[0]], [center[1]], c='red', s=100, marker='*', 
             label=f'Center {tuple(center)}')
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'2D Samples from Discrete Gaussian (σ = {sigma})')
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add histogram subplots for marginal distributions
    # Create a divider for the axes to add the histograms
    divider = make_axes_locatable(ax)
    
    # Add histogram subplots for marginal distributions
    divider = plt.figure(figsize=(12, 12))
    
    # Create a main plot and two histograms
    gs = plt.GridSpec(4, 4)
    ax_main = divider.add_subplot(gs[1:4, 0:3])
    ax_histx = divider.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_histy = divider.add_subplot(gs[1:4, 3], sharey=ax_main)
    
    # Redraw the main scatter plot
    try:
        scatter = ax_main.scatter(x_coords, y_coords, c=z, alpha=0.7, 
                                cmap='viridis', s=30, edgecolor='k', linewidth=0.5)
    except:
        ax_main.scatter(x_coords, y_coords, alpha=0.7, s=30, edgecolor='k', linewidth=0.5)
    
    # Add contours
    ax_main.contour(X, Y, Z, levels=10, cmap='coolwarm', alpha=0.5)
    
    # Plot the fundamental domain if basis is provided
    if lattice_basis is not None:
        ax_main.arrow(origin[0], origin[1], v1[0], v1[1], head_width=0.2, head_length=0.3, 
                    fc='blue', ec='blue')
        ax_main.arrow(origin[0], origin[1], v2[0], v2[1], head_width=0.2, head_length=0.3, 
                    fc='green', ec='green')
        ax_main.plot([0, v1[0], v1[0]+v2[0], v2[0], 0], 
                   [0, v1[1], v1[1]+v2[1], v2[1], 0], 
                   'r--', alpha=0.7)
    
    # Mark the center
    ax_main.scatter([center[0]], [center[1]], c='red', s=100, marker='*')
    
    ax_main.set_xlabel('Dimension 1')
    ax_main.set_ylabel('Dimension 2')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_aspect('equal')
    
    # Make some labels invisible
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histy.tick_params(axis='y', labelleft=False)
    
    # Add histograms
    bins = min(50, int(np.sqrt(len(samples))))
    hist_x, bins_x, _ = ax_histx.hist(x_coords, bins=bins, density=True, alpha=0.7)
    hist_y, bins_y, _ = ax_histy.hist(y_coords, bins=bins, density=True, alpha=0.7, orientation='horizontal')
    
    # Add theoretical marginal Gaussian
    x_vals = np.linspace(x_min, x_max, 1000)
    y_vals = np.linspace(y_min, y_max, 1000)
    
    # Compute normalized theoretical densities for visualization
    x_density = np.array([discrete_gaussian_pdf(x, sigma, center[0]) for x in x_vals])
    x_density = x_density / np.sum(x_density) * len(x_vals) / (x_max-x_min)
    
    y_density = np.array([discrete_gaussian_pdf(y, sigma, center[1]) for y in y_vals])
    y_density = y_density / np.sum(y_density) * len(y_vals) / (y_max-y_min)
    
    ax_histx.plot(x_vals, x_density, 'r-', alpha=0.7, label='Theoretical')
    ax_histy.plot(y_density, y_vals, 'r-', alpha=0.7)
    
    # If we have theoretical distribution data, add KL divergence or TV distance
    if theoretical_dist:
        # Compute empirical distribution
        sample_counts = {}
        for sample in samples:
            sample_tuple = tuple(map(float, sample))
            sample_counts[sample_tuple] = sample_counts.get(sample_tuple, 0) + 1
        
        # Normalize
        total_samples = len(samples)
        empirical_probs = {k: v / total_samples for k, v in sample_counts.items()}
        
        # Compute KL divergence (only for points in both distributions)
        common_points = set(empirical_probs.keys()) & set(theoretical_dist.keys())
        if common_points:
            kl = sum(empirical_probs[p] * log(empirical_probs[p] / theoretical_dist[p])
                    for p in common_points if theoretical_dist[p] > 0)
            
            # Add annotation
            ax_main.text(0.02, 0.02, 
                       f'KL(empirical || theoretical): {kl:.6f}',
                       transform=ax_main.transAxes, 
                       verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add title
    if title:
        divider.suptitle(title, fontsize=14)
    else:
        divider.suptitle(f'2D Samples from Discrete Gaussian (σ = {sigma})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/{filename}')
    plt.close()

def plot_3d_samples(samples, sigma, filename, title=None, center=None, basis=None):
    """
    Create a 3D scatter plot of the samples.
    
    3D visualization is crucial for understanding the spatial distribution of samples
    in higher dimensions. Multiple views are provided to overcome the limitations
    of static 3D plots.
    
    Args:
        samples: List of 3D samples
        sigma: The standard deviation used for the Gaussian
        filename: Filename to save the plot
        title: Optional title for the plot
        center: Center of the Gaussian distribution
        basis: Optional lattice basis to visualize
        
    Returns:
        None (saves plot to file)
    """
    if center is None:
        center = [0, 0, 0]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, and z coordinates
    x_coords = [float(sample[0]) for sample in samples]
    y_coords = [float(sample[1]) for sample in samples]
    z_coords = [float(sample[2]) for sample in samples]
    
    # Compute distance from center for coloring
    if center != [0, 0, 0]:
        distances = [np.sqrt(sum((float(sample[i]) - center[i])**2 for i in range(3))) 
                    for sample in samples]
        c_vals = distances
        cmap = 'viridis'
        clabel = 'Distance from Center'
    else:
        # Use z-coordinate for coloring if center is origin
        c_vals = z_coords
        cmap = 'viridis'
        clabel = 'Dimension 3 Value'
    
    # Create a scatter plot
    scatter = ax.scatter(x_coords, y_coords, z_coords, 
                       c=c_vals, cmap=cmap, 
                       alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
    
    # Mark the center
    ax.scatter([center[0]], [center[1]], [center[2]], 
             c='red', s=100, marker='*', label='Center')
    
    # If basis is provided, plot the fundamental parallelpiped
    if basis is not None and basis.nrows() == 3:
        # Extract basis vectors
        v1 = np.array([float(basis[0, 0]), float(basis[0, 1]), float(basis[0, 2])])
        v2 = np.array([float(basis[1, 0]), float(basis[1, 1]), float(basis[1, 2])])
        v3 = np.array([float(basis[2, 0]), float(basis[2, 1]), float(basis[2, 2])])
        
        # Create the vertices of the parallelpiped
        origin = np.zeros(3)
        vertices = [
            origin,
            v1,
            v2,
            v3,
            v1 + v2,
            v1 + v3,
            v2 + v3,
            v1 + v2 + v3
        ]
        
        # Plot edges
        edges = [
            (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
            (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)
        ]
        
        for start, end in edges:
            ax.plot([vertices[start][0], vertices[end][0]],
                  [vertices[start][1], vertices[end][1]],
                  [vertices[start][2], vertices[end][2]],
                  'r--', alpha=0.7)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'3D Samples from Discrete Gaussian (σ = {sigma})')
    
    plt.colorbar(scatter, ax=ax, label=clabel)
    
    # Add theoretical vs. empirical annotation
    if len(samples) > 100:
        # Calculate average distance from center
        mean_dist = np.mean(distances) if 'distances' in locals() else np.mean(
            [np.sqrt(sum(float(s[i])**2 for i in range(3))) for s in samples]
        )
        
        # Theoretical expected distance for a 3D Gaussian
        # For a 3D standard normal, expected distance from origin is sqrt(8/π)≈1.6σ
        expected_dist = sigma * 1.6
        
        # Add annotation
        ax.text2D(0.02, 0.02, 
                f'Mean distance: {mean_dist:.3f}\n'
                f'Expected (3D Gaussian): {expected_dist:.3f}\n'
                f'Ratio: {mean_dist/expected_dist:.3f}',
                transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save multiple views for 3D visualization
    views = [(30, 30), (0, 0), (0, 90), (90, 0)]
    for i, (elev, azim) in enumerate(views):
        ax.view_init(elev=elev, azim=azim)
        plt.savefig(f'results/plots/{filename}_view{i+1}.png')
    
    plt.close()

def plot_2d_projections(samples, sigma, filename, title=None, center=None, basis=None):
    """
    Create 2D projections of higher-dimensional samples.
    
    2D projections are essential for visualizing higher-dimensional data. In MCMC sampling,
    they help identify correlation structures between different dimensions that might
    indicate poor mixing or other issues.
    
    Args:
        samples: List of samples (3D or 4D)
        sigma: The standard deviation used for the Gaussian
        filename: Filename to save the plot
        title: Optional title for the plot
        center: Center of the Gaussian distribution
        basis: Optional lattice basis for context
        
    Returns:
        None (saves plot to file)
    """
    n_dims = len(samples[0])
    if n_dims < 3:
        print("Warning: 2D projections are only meaningful for dimensions >= 3")
        return
    
    if center is None:
        center = [0] * n_dims
    
    # Create all pairs of dimensions
    pairs = [(i, j) for i in range(n_dims) for j in range(i+1, n_dims)]
    n_pairs = len(pairs)
    
    fig = plt.figure(figsize=(5 * n_pairs, 15))
    
    # Calculate point density for each projection
    for idx, (dim1, dim2) in enumerate(pairs):
        ax = fig.add_subplot(3, n_pairs//3 + (1 if n_pairs % 3 else 0), idx+1)
        
        # Extract the coordinates for this pair of dimensions
        x_coords = [float(sample[dim1]) for sample in samples]
        y_coords = [float(sample[dim2]) for sample in samples]
        
        # Calculate point density for coloring
        try:
            xy = np.vstack([x_coords, y_coords])
            z = gaussian_kde(xy)(xy)
            
            # Create a scatter plot with density-based coloring
            scatter = ax.scatter(x_coords, y_coords, c=z, 
                               cmap='viridis', alpha=0.7, s=30, 
                               edgecolor='k', linewidth=0.5)
        except:
            # Fallback if KDE fails
            ax.scatter(x_coords, y_coords, alpha=0.7, s=30, 
                     edgecolor='k', linewidth=0.5)
        
        # Add density contours
        x_min, x_max = min(x_coords) - 1, max(x_coords) + 1
        y_min, y_max = min(y_coords) - 1, max(y_coords) + 1
        
        # Create a grid of points
        x_grid = np.linspace(x_min, x_max, 50)
        y_grid = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # If basis is provided, project it onto these dimensions
        if basis is not None and basis.nrows() == n_dims:
            # Extract the relevant 2D sub-basis for this projection
            v1 = np.array([float(basis[dim1, i]) for i in range(n_dims)])
            v2 = np.array([float(basis[dim2, i]) for i in range(n_dims)])
            
            # Plot basis vectors
            ax.arrow(0, 0, v1[dim1], v1[dim2], head_width=0.2, head_length=0.3, 
                   fc='blue', ec='blue', alpha=0.7)
            ax.arrow(0, 0, v2[dim1], v2[dim2], head_width=0.2, head_length=0.3, 
                   fc='green', ec='green', alpha=0.7)
        
        # Mark the center for this projection
        ax.scatter([center[dim1]], [center[dim2]], 
                 c='red', s=100, marker='*')
        
        ax.set_xlabel(f'Dimension {dim1+1}')
        ax.set_ylabel(f'Dimension {dim2+1}')
        ax.set_title(f'Proj: Dim {dim1+1} vs Dim {dim2+1}')
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(x_coords, y_coords)[0, 1]
        ax.text(0.02, 0.02, 
               f'Corr: {corr:.3f}',
               transform=ax.transAxes, 
               verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f'{n_dims}D Samples Projected to 2D (σ = {sigma})', fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the title
    plt.savefig(f'results/plots/{filename}')
    plt.close()

def plot_pca_projection(samples, sigma, filename, title=None, variance_threshold=0.95):
    """
    Create a PCA projection of higher-dimensional samples to 2D or 3D.
    
    PCA (Principal Component Analysis) is a powerful tool for visualizing high-dimensional
    data. For lattice-based distributions, it helps identify the principal directions
    of variance, which are related to the shortest lattice vectors.
    
    Args:
        samples: List of samples
        sigma: The standard deviation used for the Gaussian
        filename: Filename to save the plot
        title: Optional title for the plot
        variance_threshold: Threshold for cumulative explained variance
        
    Returns:
        None (saves plot to file)
    """
    n_dims = len(samples[0])
    if n_dims < 3:
        print("Warning: PCA projection to 2D is only meaningful for dimensions >= 3")
        return
    
    # Convert samples to numpy array
    samples_array = np.array([[float(x) for x in sample] for sample in samples])
    
    # Apply PCA
    from sklearn.decomposition import PCA
    
    # First, determine how many components we need to explain a certain percentage of variance
    full_pca = PCA()
    full_pca.fit(samples_array)
    
    # Compute cumulative explained variance
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    
    # Find number of components needed to explain variance_threshold of variance
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    n_components = min(3, n_components)  # Cap at 3 for visualization
    
    # Apply PCA with the determined number of components
    pca = PCA(n_components=n_components)
    samples_pca = pca.fit_transform(samples_array)
    
    # Plot the projection
    if n_components == 2:
        # 2D projection
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Calculate point density for coloring
        try:
            xy = np.vstack([samples_pca[:, 0], samples_pca[:, 1]])
            z = gaussian_kde(xy)(xy)
            
            # Create a scatter plot with density-based coloring
            scatter = ax.scatter(samples_pca[:, 0], samples_pca[:, 1], c=z, 
                               cmap='viridis', alpha=0.7, s=30, 
                               edgecolor='k', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Point Density')
        except:
            # Fallback if KDE fails
            ax.scatter(samples_pca[:, 0], samples_pca[:, 1], alpha=0.7, s=30, 
                     edgecolor='k', linewidth=0.5)
        
        # Add explained variance ratio
        ax.text(0.02, 0.98, 
               f'Explained variance:\nPC1: {pca.explained_variance_ratio_[0]:.3f}\n'
               f'PC2: {pca.explained_variance_ratio_[1]:.3f}\n'
               f'Total: {sum(pca.explained_variance_ratio_):.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'PCA Projection to 2D (σ = {sigma}, {n_dims}D → 2D)')
        
        ax.grid(True, alpha=0.3)
        
    elif n_components == 3:
        # 3D projection
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Compute distance from origin in PCA space for coloring
        distances = np.sqrt(np.sum(samples_pca**2, axis=1))
        
        scatter = ax.scatter(samples_pca[:, 0], samples_pca[:, 1], samples_pca[:, 2],
                           c=distances, cmap='viridis', alpha=0.6, s=30,
                           edgecolor='k', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Distance from Origin (PCA space)')
        
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        
        # Add explained variance annotation
        ax.text2D(0.02, 0.98, 
                f'Explained variance:\nPC1: {pca.explained_variance_ratio_[0]:.3f}\n'
                f'PC2: {pca.explained_variance_ratio_[1]:.3f}\n'
                f'PC3: {pca.explained_variance_ratio_[2]:.3f}\n'
                f'Total: {sum(pca.explained_variance_ratio_):.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'PCA Projection to 3D (σ = {sigma}, {n_dims}D → 3D)')
        
        # Save multiple views for 3D visualization
        views = [(30, 30), (0, 0), (0, 90), (90, 0)]
        for i, (elev, azim) in enumerate(views):
            ax.view_init(elev=elev, azim=azim)
            plt.savefig(f'results/plots/{filename}_view{i+1}.png')
    
    # For both 2D and 3D, also save the standard view
    plt.tight_layout()
    plt.savefig(f'results/plots/{filename}')
    plt.close()
    
    # Return the PCA object for potential further analysis
    return pca

# ----------------------------------------
# Statistical Distance Measures
# ----------------------------------------

def compute_total_variation_distance(samples, sigma, lattice_basis, center=None, exact=False, radius=3):
    """
    Compute the total variation distance between the empirical
    distribution of the samples and the ideal discrete Gaussian.
    
    For small dimensions, this can optionally compute the exact TV distance by
    enumerating all lattice points within a reasonable radius.
    
    Args:
        samples: List of samples
        sigma: The standard deviation used for the Gaussian
        lattice_basis: The lattice basis
        center: Center of the Gaussian distribution
        exact: Whether to compute the exact TV distance (only for small dimensions)
        radius: Radius (in standard deviations) to consider for exact computation
        
    Returns:
        The estimated total variation distance and additional info
    """
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
    
    if exact and n_dims <= 3:
        # For small dimensions, compute the exact TV distance
        # by enumerating all lattice points within a reasonable radius
        
        # Determine the range of lattice points to consider
        # We'll enumerate all integer combinations in a box and map to the lattice
        lattice_points = set()
        theoretical_probs = {}
        Z = 0  # Normalization constant
        
        # For each integer combination in a box of size [-radius*sigma, radius*sigma]^n
        from itertools import product
        
        # Inverse of the lattice basis for converting to lattice coordinates
        try:
            B_inv = lattice_basis.inverse()
        except:
            # If the basis is not invertible, use the pseudo-inverse
            B_np = np.array([[float(lattice_basis[i,j]) for j in range(n_dims)] 
                           for i in range(n_dims)])
            B_inv_np = np.linalg.pinv(B_np)
            B_inv = matrix(RR, B_inv_np)
        
        grid_min = -int(radius * sigma)
        grid_max = int(radius * sigma) + 1
        grid_range = range(grid_min, grid_max)
        
        # Convert center to lattice coordinates
        center_lattice = B_inv * center
        
        # Enumerate all lattice points in the box
        for coords in product(grid_range, repeat=n_dims):
            # Convert to lattice coordinates
            lattice_coords = vector(ZZ, coords)
            
            # Convert to original coordinates
            lattice_point = lattice_basis * lattice_coords
            
            # Compute density
            density = discrete_gaussian_pdf(lattice_point, sigma, center)
            
            # Skip points with negligible density
            if density < 1e-10:
                continue
            
            # Add to the set of lattice points
            lattice_point_tuple = tuple(map(float, lattice_point))
            lattice_points.add(lattice_point_tuple)
            
            # Accumulate normalization constant
            Z += density
        
        # Compute theoretical probabilities
        for point in lattice_points:
            point_vector = vector(RR, point)
            density = discrete_gaussian_pdf(point_vector, sigma, center)
            theoretical_probs[point] = density / Z
        
        # Now compute the total variation distance
        # TV = (1/2) * sum_{x} |P(x) - Q(x)|
        tv_distance = 0
        all_points = set(empirical_probs.keys()) | set(theoretical_probs.keys())
        
        for point in all_points:
            emp_prob = empirical_probs.get(point, 0)
            theo_prob = theoretical_probs.get(point, 0)
            tv_distance += abs(emp_prob - theo_prob)
        
        tv_distance *= 0.5
        
        return {
            'tv_distance': tv_distance,
            'empirical_probs': empirical_probs,
            'theoretical_probs': theoretical_probs,
            'Z': Z,
            'method': 'exact',
            'radius': radius
        }
    
    else:
        # For larger dimensions, use the standard approximation
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
        
        return {
            'tv_distance': tv_distance,
            'empirical_probs': empirical_probs,
            'theoretical_probs': theoretical_probs,
            'Z': Z,
            'method': 'approximate',
            'radius': None
        }

def compute_kl_divergence(samples, sigma, lattice_basis, center=None, exact=False, radius=3):
    """
    Compute the KL divergence between the empirical
    distribution of the samples and the ideal discrete Gaussian.
    
    KL divergence measures the information loss when using one distribution
    to approximate another. It is a more sensitive measure than TV distance.
    
    Args:
        samples: List of samples
        sigma: The standard deviation used for the Gaussian
        lattice_basis: The lattice basis
        center: Center of the Gaussian distribution
        exact: Whether to compute based on the exact theoretical distribution
        radius: Radius (in standard deviations) to consider for exact computation
        
    Returns:
        The estimated KL divergence and additional info
    """
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

# ----------------------------------------
# Convergence Analysis Functions
# ----------------------------------------

def compute_theoretical_spectral_bound(B, sigma, center=None, radius=6):
    """
    Compute a theoretical bound on the spectral gap for small dimensions.
    
    This implements the bound from Theorem 2.1, which relates the spectral gap δ
    to the ratio of partition functions: δ ≥ ∏ᵢ₌₁ⁿ ρ_σᵢ(Z) / ρ_σ,c(Λ)
    
    Args:
        B: The lattice basis matrix
        sigma: The standard deviation of the Gaussian
        center: Center of the Gaussian (default: origin)
        radius: Radius in standard deviations to consider
        
    Returns:
        Estimated lower bound on the spectral gap
    """
    n = B.nrows()
    
    if n > 3:
        print("Warning: Theoretical spectral bound computation is only feasible for dimensions ≤ 3")
        return None
    
    if center is None:
        center = vector(RR, [0] * n)
    
    # Compute orthogonalized basis using Gram-Schmidt
    B_copy = matrix(QQ, B)
    GSO = B_copy.gram_schmidt()
    Q = GSO[0]  # Orthogonal basis
    
    # Compute bound on the integer range to consider
    bound = int(np.ceil(radius * sigma))
    
    # Compute ρ_σ,c(Λ) = ∑_{x ∈ Λ} exp(-||x-c||²/2σ²)
    rho_lambda = 0
    
    # For small dimensions, we can enumerate lattice points
    from itertools import product
    for coords in product(range(-bound, bound + 1), repeat=n):
        x = vector(ZZ, coords)
        lattice_point = B * x
        density = discrete_gaussian_pdf(lattice_point, sigma, center)
        rho_lambda += density
    
    # Compute ∏ᵢ₌₁ⁿ ρ_σᵢ(Z) = ∏ᵢ₌₁ⁿ ∑_{k ∈ Z} exp(-k²/2σᵢ²)
    rho_product = 1
    
    for i in range(n):
        b_i_star = vector(RR, Q.row(i))
        b_star_norm = b_i_star.norm()
        sigma_i = sigma / b_star_norm
        
        # Compute ρ_σᵢ(Z) = ∑_{k ∈ Z} exp(-k²/2σᵢ²)
        rho_zi = 0
        for k in range(-bound, bound + 1):
            rho_zi += discrete_gaussian_pdf(k, sigma_i, 0)
        
        rho_product *= rho_zi
    
    # Compute the spectral gap bound: δ ≥ ∏ᵢ₌₁ⁿ ρ_σᵢ(Z) / ρ_σ,c(Λ)
    if rho_lambda > 0:
        spectral_bound = rho_product / rho_lambda
    else:
        spectral_bound = 0
    
    return spectral_bound

def estimate_mixing_time(B, sigma, max_iterations=10000, threshold=0.1, step_size=100, c=None, num_reference=5000):
    """
    Estimate the mixing time of the IMHK algorithm empirically.
    
    The mixing time is defined as the number of iterations needed for the TV distance
    to the stationary distribution to drop below a threshold (typically 0.1).
    
    Args:
        B: The lattice basis matrix
        sigma: The standard deviation of the Gaussian
        max_iterations: Maximum number of iterations to run
        threshold: TV distance threshold for convergence
        step_size: Interval for checking convergence
        c: Center of the Gaussian (default: origin)
        num_reference: Number of reference samples
        
    Returns:
        Estimated mixing time and TV distance trajectory
    """
    n = B.nrows()
    if c is None:
        c = vector(RR, [0] * n)
    
    # Generate reference samples using Klein's algorithm
    # These serve as an approximation to the target distribution
    print(f"Generating {num_reference} reference samples...")
    reference_samples = [klein_sampler(B, sigma, c) for _ in range(num_reference)]
    
    # Run IMHK sampler and track TV distance
    print("Running IMHK sampler to estimate mixing time...")
    samples, _, all_samples, _ = imhk_sampler(B, sigma, max_iterations, c, burn_in=0)
    
    # Compute TV distance at regular intervals
    tv_distances = []
    mixing_time = max_iterations
    
    for t in range(step_size, max_iterations + 1, step_size):
        current_samples = all_samples[:t]
        
        # Compute empirical TV distance
        tv_result = compute_total_variation_distance(current_samples, sigma, B, c)
        tv_dist = tv_result['tv_distance']
        
        tv_distances.append((t, tv_dist))
        print(f"  Iteration {t}: TV distance = {tv_dist:.6f}")
        
        # Check if we've reached the threshold
        if tv_dist < threshold and mixing_time == max_iterations:
            mixing_time = t
            print(f"  Mixing time threshold reached at iteration {t}")
    
    # If we didn't reach the threshold, estimate using exponential decay fit
    if mixing_time == max_iterations and len(tv_distances) > 3:
        # Extract iterations and TV distances
        iterations, tv_values = zip(*tv_distances)
        iterations = np.array(iterations)
        tv_values = np.array(tv_values)
        
        # Fit exponential decay: TV(t) ≈ a * exp(-b * t)
        try:
            # Take log of TV values for linear fit
            log_tv = np.log(tv_values)
            valid_idx = np.isfinite(log_tv)
            
            if np.sum(valid_idx) > 2:  # Need at least 3 points for fitting
                # Fit line to log(TV) vs iterations
                coeffs = np.polyfit(iterations[valid_idx], log_tv[valid_idx], 1)
                decay_rate = -coeffs[0]  # Negative slope gives decay rate
                
                if decay_rate > 0:
                    # Estimate mixing time: t_mix ≈ log(1/threshold) / decay_rate
                    estimated_mixing_time = int(np.ceil(np.log(1/threshold) / decay_rate))
                    print(f"  Estimated mixing time from decay rate: {estimated_mixing_time}")
                    
                    # Use this estimate if it's reasonable
                    if estimated_mixing_time < 10 * max_iterations:
                        mixing_time = estimated_mixing_time
        except:
            pass
    
    return mixing_time, tv_distances

def plot_convergence(tv_distances, mixing_time, spectral_bound=None, filename=None, title=None):
    """
    Plot the convergence behavior of the IMHK sampler.
    
    Args:
        tv_distances: List of (iteration, TV distance) pairs
        mixing_time: Estimated mixing time
        spectral_bound: Optional theoretical spectral gap bound
        filename: Filename to save the plot
        title: Optional title for the plot
        
    Returns:
        None (saves plot to file)
    """
    plt.figure(figsize=(10, 6))
    
    # Extract iterations and TV distances
    iterations, tv_values = zip(*tv_distances)
    
    # Plot TV distance trajectory
    plt.plot(iterations, tv_values, 'o-', label='Empirical TV Distance')
    
    # Add threshold line
    plt.axhline(0.1, color='r', linestyle='--', label='ε = 0.1 Threshold')
    
    # Mark the estimated mixing time
    plt.axvline(mixing_time, color='g', linestyle='--', 
                label=f'Mixing Time: {mixing_time}')
    
    # If we have a spectral bound, add theoretical mixing time
    if spectral_bound is not None and spectral_bound > 0:
        # Theoretical mixing time: t_mix ≤ log(1/ε)/δ
        theoretical_mixing_time = np.ceil(np.log(10) / spectral_bound)  # For ε = 0.1
        
        plt.axvline(theoretical_mixing_time, color='purple', linestyle=':',
                   label=f'Theoretical Bound: {theoretical_mixing_time:.0f}')
    
    # Add exponential decay fit
    if len(tv_values) > 3:
        try:
            # Fit exponential decay
            log_tv = np.log(tv_values)
            valid_idx = np.isfinite(log_tv)
            
            if np.sum(valid_idx) > 2:
                # Convert to numpy arrays for fitting
                x_data = np.array(iterations)[valid_idx]
                y_data = log_tv[valid_idx]
                
                # Fit line to log(TV) vs iterations
                coeffs = np.polyfit(x_data, y_data, 1)
                decay_rate = -coeffs[0]  # Negative slope
                
                if decay_rate > 0:
                    # Generate fit line
                    x_fit = np.linspace(min(iterations), max(iterations), 100)
                    y_fit = np.exp(coeffs[1]) * np.exp(-decay_rate * x_fit)
                    
                    plt.plot(x_fit, y_fit, 'r-', alpha=0.5,
                           label=f'Exp. Decay (rate={decay_rate:.5f})')
                    
                    # Add text annotation
                    plt.text(0.7, 0.9, 
                           f"Decay Rate: {decay_rate:.5f}\n"
                           f"Half-life: {np.log(2)/decay_rate:.1f} iterations",
                           transform=plt.gca().transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        except:
            pass
    
    plt.xlabel('Iterations')
    plt.ylabel('Total Variation Distance')
    plt.yscale('log')  # Log scale often better shows convergence
    
    if title:
        plt.title(title)
    else:
        plt.title('IMHK Convergence Analysis')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if filename:
        plt.tight_layout()
        plt.savefig(f'results/plots/{filename}')
        plt.close()
    else:
        plt.show()

# ----------------------------------------
# Performance Profiling and Optimization
# ----------------------------------------

def profile_sampler(B, sigma, num_samples=1000, c=None):
    """
    Profile the IMHK sampler to identify bottlenecks.
    
    Args:
        B: The lattice basis matrix
        sigma: The standard deviation of the Gaussian
        num_samples: Number of samples to generate
        c: Center of the Gaussian
        
    Returns:
        Dictionary of profiling results
    """
    import cProfile
    import io
    import pstats
    
    # Create a string buffer to capture the profiling output
    s = io.StringIO()
    
    # Profile the IMHK sampler
    profile = cProfile.Profile()
    profile.enable()
    
    # Run the sampler
    samples, acceptance_rate, _, _ = imhk_sampler(B, sigma, num_samples, c, burn_in=num_samples//10)
    
    profile.disable()
    
    # Get the profiling statistics
    ps = pstats.Stats(profile, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions by cumulative time
    
    # Extract the profiling output
    profile_output = s.getvalue()
    
    # Save the profiling results to a file
    with open(f'results/profiles/imhk_profile_dim{B.nrows()}_sigma{sigma}.txt', 'w') as f:
        f.write(profile_output)
    
    return {
        'samples': samples,
        'acceptance_rate': acceptance_rate,
        'profile_output': profile_output
    }

def optimize_discrete_gaussian_pdf(vectorized=True):
    """
    Replace the discrete_gaussian_pdf function with an optimized version.
    
    Args:
        vectorized: Whether to use vectorized operations
        
    Returns:
        Original function for reference
    """
    # Store the original function
    original_function = discrete_gaussian_pdf
    
    if vectorized:
        # Define a vectorized version using NumPy
        def vectorized_discrete_gaussian_pdf(x, sigma, center=None):
            """
            Vectorized implementation of discrete Gaussian PDF using NumPy.
            
            Args:
                x: The point to evaluate (can be a vector or scalar)
                sigma: The standard deviation of the Gaussian
                center: The center of the Gaussian (default: origin)
                
            Returns:
                The probability density at point x
            """
            # Handle small sigma case
            if sigma < 0.001:
                # For very small sigma, the distribution concentrates at the center
                if center is None:
                    center = np.zeros(len(x) if hasattr(x, '__len__') else 1)
                
                # Return 1 if x is the closest lattice point to center, 0 otherwise
                if hasattr(x, '__len__'):
                    closest = np.round(center)
                    if np.allclose(x, closest):
                        return 1.0
                    return 0.0
                else:
                    closest = round(center)
                    return 1.0 if x == closest else 0.0
            
            # Normal case
            if center is None:
                if hasattr(x, '__len__'):
                    center = np.zeros(len(x))
                else:
                    center = 0
            
            # Convert inputs to NumPy arrays for vectorized computation
            if hasattr(x, '__len__'):
                if isinstance(x, (list, tuple)):
                    x = np.array(x, dtype=float)
                elif isinstance(x, sage.modules.vector_real_dense.Vector_real_dense):
                    x = np.array([float(xi) for xi in x], dtype=float)
                
                if isinstance(center, (list, tuple)):
                    center = np.array(center, dtype=float)
                elif isinstance(center, sage.modules.vector_real_dense.Vector_real_dense):
                    center = np.array([float(ci) for ci in center], dtype=float)
                
                # Compute squared norm
                squared_norm = np.sum((x - center) ** 2)
            else:
                # Scalar case
                squared_norm = (x - center) ** 2
            
            # Compute the density
            return np.exp(-squared_norm / (2 * sigma ** 2))
        
        # Replace the original function with the vectorized version
        globals()['discrete_gaussian_pdf'] = vectorized_discrete_gaussian_pdf
    
    return original_function

def parallel_imhk_sampler(B, sigma, num_samples, c=None, burn_in=1000, num_chains=4, num_processes=None):
    """
    Run multiple IMHK chains in parallel for better performance.
    
    Args:
        B: The lattice basis matrix
        sigma: The standard deviation of the Gaussian
        num_samples: Number of samples per chain
        c: Center of the Gaussian
        burn_in: Number of burn-in samples per chain
        num_chains: Number of chains to run
        num_processes: Number of processes (default: number of CPU cores)
        
    Returns:
        Combined samples and diagnostics
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    # Create a process pool
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Define the function to run a single chain
        def run_chain(seed):
            # Set a different random seed for each chain
            np.random.seed(seed)
            set_random_seed(seed)
            
            # Run the IMHK sampler
            return imhk_sampler(B, sigma, num_samples, c, burn_in)
        
        # Run chains in parallel
        results = pool.map(run_chain, range(1000, 1000 + num_chains))
    
    # Extract samples and diagnostics
    all_samples = []
    chain_samples = []
    acceptance_rates = []
    
    for samples, acceptance_rate, all_chain_samples, all_accepts in results:
        all_samples.extend(samples)
        chain_samples.append(samples)
        acceptance_rates.append(acceptance_rate)
    
    return all_samples, chain_samples, acceptance_rates

# ----------------------------------------
# Visualization Enhancement
# ----------------------------------------

def set_publication_style():
    """
    Set matplotlib style for publication-quality plots.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
        'lines.linewidth': 2,
        'lines.markersize': 6,
    })
    
    # Uncomment for LaTeX rendering if available
    # plt.rcParams.update({
    #     'text.usetex': True,
    #     'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
    # })

def generate_summary_table(results, filename):
    """
    Generate a publication-ready LaTeX table summarizing results.
    
    Args:
        results: Dictionary of experimental results
        filename: Base filename for the output
        
    Returns:
        None (saves table to file)
    """
    # Ensure the logs directory exists
    os.makedirs('results/logs', exist_ok=True)
    
    # Create a basic table for dimensions, sigma, basis type, TV distance, etc.
    with open(f'results/logs/{filename}_summary.tex', 'w') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Summary of IMHK Sampler Performance}' + '\n')
        f.write(r'\begin{tabular}{cccccc}' + '\n')
        f.write(r'\toprule' + '\n')
        f.write(r'Dim & $\sigma$ & Basis & TV Distance & Acceptance Rate & Mixing Time \\' + '\n')
        f.write(r'\midrule' + '\n')
        
        # Sort results by dimension, then sigma
        keys = sorted(results.keys(), key=lambda k: (k[0], k[1]))
        
        for key in keys:
            dim, sigma, basis_type, center_tuple = key
            result = results[key]
            
            # Extract relevant metrics
            tv_distance = result.get('imhk_tv_distance', 'N/A')
            acceptance_rate = result.get('imhk_acceptance_rate', 'N/A')
            mixing_time = result.get('mixing_time', 'N/A')
            
            # Format the values
            if isinstance(tv_distance, (int, float)):
                tv_str = f'{tv_distance:.4f}'
            else:
                tv_str = tv_distance
                
            if isinstance(acceptance_rate, (int, float)):
                acc_str = f'{acceptance_rate:.3f}'
            else:
                acc_str = acceptance_rate
                
            if isinstance(mixing_time, (int, float)):
                mix_str = f'{mixing_time}'
            else:
                mix_str = mixing_time
            
            # Write the table row
            f.write(f'{dim} & {sigma:.1f} & {basis_type} & {tv_str} & {acc_str} & {mix_str} \\\\' + '\n')
        
        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\label{tab:imhk_summary}' + '\n')
        f.write(r'\end{table}' + '\n')
    
    # Create more detailed tables for specific analyses
    # Table for spectral gap and convergence comparisons
    with open(f'results/logs/{filename}_convergence.tex', 'w') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Convergence Properties of IMHK Sampler}' + '\n')
        f.write(r'\begin{tabular}{ccccc}' + '\n')
        f.write(r'\toprule' + '\n')
        f.write(r'Dim & $\sigma$ & Basis & Spectral Gap & Convergence Rate \\' + '\n')
        f.write(r'\midrule' + '\n')
        
        # Filter keys for which we have spectral gap information
        convergence_keys = [k for k in keys if 'spectral_gap' in results[k]]
        
        for key in convergence_keys:
            dim, sigma, basis_type, _ = key
            result = results[key]
            
            spectral_gap = result.get('spectral_gap', 'N/A')
            conv_rate = result.get('convergence_rate', 'N/A')
            
            # Format values
            if isinstance(spectral_gap, (int, float)):
                gap_str = f'{spectral_gap:.6f}'
            else:
                gap_str = spectral_gap
                
            if isinstance(conv_rate, (int, float)):
                rate_str = f'{conv_rate:.6f}'
            else:
                rate_str = conv_rate
            
            f.write(f'{dim} & {sigma:.1f} & {basis_type} & {gap_str} & {rate_str} \\\\' + '\n')
        
        f.write(r'\bottomrule' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\label{tab:imhk_convergence}' + '\n')
        f.write(r'\end{table}' + '\n')
        
    print(f"LaTeX tables generated in results/logs/{filename}_*.tex")

# ----------------------------------------
# Main Execution Functions
# ----------------------------------------

def run_experiment(dim, sigma, num_samples, basis_type='identity', compare_with_klein=True, center=None):
    """
    Run a complete experiment with IMHK sampling and analysis.
    
    Args:
        dim: The dimension of the lattice (2, 3, or 4)
        sigma: The standard deviation of the Gaussian
        num_samples: The number of samples to generate
        basis_type: The type of lattice basis to use ('identity', 'skewed', 'ill-conditioned')
        compare_with_klein: Whether to compare with Klein's algorithm
        center: Center of the Gaussian distribution
        
    Returns:
        A dictionary of results
    """
    # Set up the center
    if center is None:
        center = vector(RR, [0] * dim)
    else:
        center = vector(RR, center)
    
    # Create the lattice basis
    if basis_type == 'identity':
        B = matrix.identity(RR, dim)
    elif basis_type == 'skewed':
        B = matrix.identity(RR, dim)
        B[0, 1] = 1.5  # Add some skew
        if dim >= 3:
            B[0, 2] = 0.5
    elif basis_type == 'ill-conditioned':
        B = matrix.identity(RR, dim)
        # Make the matrix ill-conditioned
        B[0, 0] = 10.0
        B[1, 1] = 0.1
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")
    
    experiment_name = f"dim{dim}_sigma{sigma}_{basis_type}"
    if any(c != 0 for c in center):
        experiment_name += f"_center{'_'.join(str(c) for c in center)}"
    
    # Calculate smoothing parameter for reference
    # For an identity basis, the smoothing parameter η_ε(Λ) ≈ sqrt(ln(2n/ε)/π)
    epsilon = 0.01  # A small constant, typically 2^(-n) for some n
    smoothing_param = sqrt(log(2*dim/epsilon)/pi)
    
    print(f"Running experiment: dim={dim}, sigma={sigma}, basis={basis_type}")
    print(f"Smoothing parameter η_{epsilon}(Λ) ≈ {smoothing_param:.4f} for reference")
    print(f"σ/η ratio: {sigma/smoothing_param:.4f}")
    
    # Run IMHK sampler
    burn_in = min(5000, num_samples)  # Use appropriate burn-in
    start_time = time.time()
    imhk_samples, acceptance_rate, all_samples, all_accepts = imhk_sampler(
        B, sigma, num_samples, center, burn_in=burn_in)
    imhk_time = time.time() - start_time
    
    # Run Klein sampler for comparison if requested
    if compare_with_klein:
        start_time = time.time()
        klein_samples = [klein_sampler(B, sigma, center) for _ in range(num_samples)]
        klein_time = time.time() - start_time
    
    # Analyze acceptance rate over time
    plot_acceptance_trace(all_accepts, f"acceptance_trace_{experiment_name}.png")
    
    # Run diagnostics on IMHK samples
    # Trace plots
    plot_trace(imhk_samples, f"trace_imhk_{experiment_name}.png", 
             f"IMHK Sample Trace (σ={sigma}, {basis_type} basis)")
    
    # Autocorrelation
    acf_by_dim = compute_autocorrelation(imhk_samples)
    plot_autocorrelation(acf_by_dim, f"acf_imhk_{experiment_name}.png", 
                       f"IMHK Autocorrelation (σ={sigma}, {basis_type} basis)")
    
    # Effective Sample Size
    ess_values = compute_ess(imhk_samples)
    
    # Visualization
    if dim == 2:
        plot_2d_samples(imhk_samples, sigma, f"samples_imhk_{experiment_name}.png", 
                      B, f"IMHK Samples (σ={sigma}, {basis_type} basis)", center)
        if compare_with_klein:
            plot_2d_samples(klein_samples, sigma, f"samples_klein_{experiment_name}.png", 
                          B, f"Klein Samples (σ={sigma}, {basis_type} basis)", center)
    elif dim == 3:
        plot_3d_samples(imhk_samples, sigma, f"samples_imhk_{experiment_name}", 
                      f"IMHK Samples (σ={sigma}, {basis_type} basis)", center)
        if compare_with_klein:
            plot_3d_samples(klein_samples, sigma, f"samples_klein_{experiment_name}", 
                          f"Klein Samples (σ={sigma}, {basis_type} basis)", center)
    
    # For higher dimensions, create 2D projections
    if dim >= 3:
        plot_2d_projections(imhk_samples, sigma, f"projections_imhk_{experiment_name}.png", 
                          f"IMHK Projections (σ={sigma}, {basis_type} basis)", center)
        if compare_with_klein:
            plot_2d_projections(klein_samples, sigma, f"projections_klein_{experiment_name}.png", 
                              f"Klein Projections (σ={sigma}, {basis_type} basis)", center)
    
    # For all dimensions, create PCA projection to 2D
    plot_pca_projection(imhk_samples, sigma, f"pca_imhk_{experiment_name}.png", 
                      f"IMHK PCA Projection (σ={sigma}, {basis_type} basis)")
    if compare_with_klein:
        plot_pca_projection(klein_samples, sigma, f"pca_klein_{experiment_name}.png", 
                          f"Klein PCA Projection (σ={sigma}, {basis_type} basis)")
    
    # Compute statistical distances
    tv_distance = compute_total_variation_distance(imhk_samples, sigma, B, center)
    
    # Compute KL divergence for small dimensions
    kl_divergence = None
    if dim <= 3:  # Only compute for small dimensions due to computational complexity
        kl_divergence = compute_kl_divergence(imhk_samples, sigma, B, center)
    
    # Compile results
    results = {
        'dimension': dim,
        'sigma': sigma,
        'basis_type': basis_type,
        'center': center,
        'smoothing_parameter': smoothing_param,
        'sigma_smoothing_ratio': sigma/smoothing_param,
        'num_samples': num_samples,
        'burn_in': burn_in,
        'imhk_acceptance_rate': acceptance_rate,
        'imhk_time': imhk_time,
        'imhk_ess': ess_values,
        'imhk_tv_distance': tv_distance,
        'imhk_kl_divergence': kl_divergence
    }
    
    if compare_with_klein:
        results['klein_time'] = klein_time
        
        # Compute TV distance for Klein samples
        klein_tv_distance = compute_total_variation_distance(klein_samples, sigma, B, center)
        results['klein_tv_distance'] = klein_tv_distance
        
        # Compute KL divergence for Klein samples if feasible
        if dim <= 3:
            klein_kl_divergence = compute_kl_divergence(klein_samples, sigma, B, center)
            results['klein_kl_divergence'] = klein_kl_divergence
    
    # Log results
    with open(f"results/logs/experiment_{experiment_name}.txt", "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Dimension: {dim}\n")
        f.write(f"Sigma: {sigma}\n")
        f.write(f"Basis type: {basis_type}\n")
        f.write(f"Center: {center}\n")
        f.write(f"Smoothing parameter η_{epsilon}(Λ): {smoothing_param:.6f}\n")
        f.write(f"σ/η ratio: {sigma/smoothing_param:.6f}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Burn-in: {burn_in}\n")
        f.write("\n=== IMHK Results ===\n")
        f.write(f"Acceptance rate: {acceptance_rate:.6f}\n")
        f.write(f"Sampling time: {imhk_time:.6f} seconds\n")
        f.write(f"Effective Sample Size: {ess_values}\n")
        f.write(f"Total Variation distance: {tv_distance:.6f}\n")
        
        if kl_divergence is not None:
            f.write(f"KL divergence: {kl_divergence:.6f}\n")
        
        if compare_with_klein:
            f.write("\n=== Klein Sampler Results ===\n")
            f.write(f"Sampling time: {klein_time:.6f} seconds\n")
            f.write(f"Total Variation distance: {klein_tv_distance:.6f}\n")
            
            if dim <= 3 and 'klein_kl_divergence' in results:
                f.write(f"KL divergence: {results['klein_kl_divergence']:.6f}\n")
            
            f.write("\n=== Comparison ===\n")
            f.write(f"IMHK/Klein time ratio: {imhk_time/klein_time:.6f}\n")
            f.write(f"IMHK/Klein TV distance ratio: {tv_distance/klein_tv_distance:.6f}\n")
            
            if dim <= 3 and 'klein_kl_divergence' in results:
                f.write(f"IMHK/Klein KL divergence ratio: {kl_divergence/results['klein_kl_divergence']:.6f}\n")
    
    # Save all data for later analysis
    with open(f"results/logs/experiment_{experiment_name}.pickle", "wb") as f:
        pickle.dump(results, f)
    
    return results

def parameter_sweep(dimensions=None, sigmas=None, basis_types=None, centers=None, num_samples=1000):
    """
    Perform a parameter sweep across different dimensions, sigmas, basis types, and centers.
    
    Args:
        dimensions: List of dimensions to test (default: [2, 3, 4])
        sigmas: List of sigma values to test (default: [0.5, 1.0, 2.0, 5.0])
        basis_types: List of basis types to test (default: ['identity', 'skewed', 'ill-conditioned'])
        centers: List of centers to test (default: [[0, ..., 0]])
        num_samples: Number of samples to generate for each configuration
        
    Returns:
        A dictionary of results indexed by configuration
    """
    if dimensions is None:
        dimensions = [2, 3, 4]
    
    if sigmas is None:
        sigmas = [0.5, 1.0, 2.0, 5.0]
    
    if basis_types is None:
        basis_types = ['identity', 'skewed', 'ill-conditioned']
    
    if centers is None:
        centers = {dim: [vector(RR, [0] * dim)] for dim in dimensions}
    elif isinstance(centers, list):
        # If centers is a list of vectors, assume it applies to all dimensions
        centers = {dim: [vector(RR, c) for c in centers if len(c) == dim] for dim in dimensions}
    
    results = {}
    
    # Create a summary file
    with open("results/logs/parameter_sweep_summary.txt", "w") as summary_file:
        summary_file.write("Parameter Sweep Summary\n")
        summary_file.write("=====================\n\n")
        
        # Loop over all combinations
        for dim in dimensions:
            for sigma in sigmas:
                for basis_type in basis_types:
                    for center in centers.get(dim, [vector(RR, [0] * dim)]):
                        config_key = (dim, sigma, basis_type, tuple(center))
                        
                        # Run the experiment
                        result = run_experiment(
                            dim=dim, 
                            sigma=sigma, 
                            num_samples=num_samples,
                            basis_type=basis_type,
                            compare_with_klein=True,
                            center=center
                        )
                        
                        results[config_key] = result
                        
                        # Log summary information
                        summary_file.write(f"Configuration: dim={dim}, sigma={sigma}, ")
                        summary_file.write(f"basis={basis_type}, center={center}\n")
                        summary_file.write(f"IMHK Acceptance Rate: {result['imhk_acceptance_rate']:.4f}\n")
                        summary_file.write(f"IMHK Total Variation Distance: {result['imhk_tv_distance']:.6f}\n")
                        summary_file.write(f"Klein Total Variation Distance: {result['klein_tv_distance']:.6f}\n")
                        summary_file.write(f"IMHK/Klein TV Ratio: {result['imhk_tv_distance']/result['klein_tv_distance']:.4f}\n")
                        summary_file.write("---\n\n")
    
    # Generate comparative plots
    plot_parameter_sweep_results(results, dimensions, sigmas, basis_types)
    
    return results

def plot_parameter_sweep_results(results, dimensions, sigmas, basis_types):
    """
    Create comparative plots for the parameter sweep results.
    
    Args:
        results: Dictionary of results from parameter_sweep
        dimensions: List of dimensions tested
        sigmas: List of sigma values tested
        basis_types: List of basis types tested
        
    Returns:
        None (saves plots to files)
    """
    # Plot acceptance rate vs. sigma for each dimension and basis type
    for dim in dimensions:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for basis_type in basis_types:
            # Extract data for this dimension and basis type
            x_data = []
            y_data = []
            
            for sigma in sigmas:
                key = (dim, sigma, basis_type, tuple([0] * dim))
                if key in results:
                    x_data.append(sigma)
                    y_data.append(results[key]['imhk_acceptance_rate'])
            
            if x_data:
                ax.plot(x_data, y_data, 'o-', label=f"{basis_type}")
        
        ax.set_xlabel('Sigma')
        ax.set_ylabel('IMHK Acceptance Rate')
        ax.set_title(f'Acceptance Rate vs. Sigma (Dimension {dim})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/plots/acceptance_vs_sigma_dim{dim}.png')
        plt.close()
    
    # Plot TV distance vs. sigma for each dimension and basis type
    for dim in dimensions:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for basis_type in basis_types:
            # Extract data for this dimension and basis type
            x_data = []
            y_imhk = []
            y_klein = []
            
            for sigma in sigmas:
                key = (dim, sigma, basis_type, tuple([0] * dim))
                if key in results:
                    x_data.append(sigma)
                    y_imhk.append(results[key]['imhk_tv_distance'])
                    y_klein.append(results[key]['klein_tv_distance'])
            
            if x_data:
                ax.plot(x_data, y_imhk, 'o-', label=f"IMHK {basis_type}")
                ax.plot(x_data, y_klein, 's--', label=f"Klein {basis_type}")
        
        ax.set_xlabel('Sigma')
        ax.set_ylabel('Total Variation Distance')
        ax.set_title(f'TV Distance vs. Sigma (Dimension {dim})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/plots/tv_distance_vs_sigma_dim{dim}.png')
        plt.close()
    
    # Plot TV distance ratio (IMHK/Klein) vs. sigma
    for dim in dimensions:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for basis_type in basis_types:
            # Extract data for this dimension and basis type
            x_data = []
            y_ratio = []
            
            for sigma in sigmas:
                key = (dim, sigma, basis_type, tuple([0] * dim))
                if key in results:
                    x_data.append(sigma)
                    ratio = results[key]['imhk_tv_distance'] / results[key]['klein_tv_distance']
                    y_ratio.append(ratio)
            
            if x_data:
                ax.plot(x_data, y_ratio, 'o-', label=f"{basis_type}")
        
        ax.set_xlabel('Sigma')
        ax.set_ylabel('TV Distance Ratio (IMHK/Klein)')
        ax.set_title(f'Quality Improvement Ratio vs. Sigma (Dimension {dim})')
        ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Equal Quality')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/plots/tv_ratio_vs_sigma_dim{dim}.png')
        plt.close()
        
def compare_convergence_times(results):
    """
    Analyze and compare convergence times across different configurations.
    
    Args:
        results: Dictionary of results from parameter_sweep
        
    Returns:
        None (saves plots to files)
    """
    # Extract dimensions, sigmas, and basis types from results
    dimensions = sorted(set(key[0] for key in results.keys()))
    sigmas = sorted(set(key[1] for key in results.keys()))
    basis_types = sorted(set(key[2] for key in results.keys()))
    
    # Group by dimension
    for dim in dimensions:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for basis_type in basis_types:
            # Extract ESS-adjusted times
            x_data = []
            y_imhk = []
            y_klein = []
            
            for sigma in sigmas:
                key = (dim, sigma, basis_type, tuple([0] * dim))
                if key in results:
                    x_data.append(sigma)
                    
                    # Compute ESS-adjusted time for IMHK
                    ess_avg = np.mean(results[key]['imhk_ess'])
                    adj_time_imhk = results[key]['imhk_time'] * results[key]['num_samples'] / ess_avg
                    y_imhk.append(adj_time_imhk)
                    
                    # Klein time (no adjustment needed as samples are independent)
                    y_klein.append(results[key]['klein_time'])
            
            if x_data:
                ax.plot(x_data, y_imhk, 'o-', label=f"IMHK {basis_type}")
                ax.plot(x_data, y_klein, 's--', label=f"Klein {basis_type}")
        
        ax.set_xlabel('Sigma')
        ax.set_ylabel('Time (seconds) per Effective Sample')
        ax.set_title(f'Convergence Time vs. Sigma (Dimension {dim})')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/plots/convergence_time_dim{dim}.png')
        plt.close()
        
        # Also plot the ratio
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for basis_type in basis_types:
            x_data = []
            y_ratio = []
            
            for sigma in sigmas:
                key = (dim, sigma, basis_type, tuple([0] * dim))
                if key in results:
                    x_data.append(sigma)
                    
                    # Compute ESS-adjusted time for IMHK
                    ess_avg = np.mean(results[key]['imhk_ess'])
                    adj_time_imhk = results[key]['imhk_time'] * results[key]['num_samples'] / ess_avg
                    
                    # Compute time ratio
                    ratio = adj_time_imhk / results[key]['klein_time']
                    y_ratio.append(ratio)
            
            if x_data:
                ax.plot(x_data, y_ratio, 'o-', label=f"{basis_type}")
        
        ax.set_xlabel('Sigma')
        ax.set_ylabel('Time Ratio (IMHK/Klein)')
        ax.set_title(f'Time Overhead Ratio vs. Sigma (Dimension {dim})')
        ax.axhline(1.0, color='r', linestyle='--', alpha=0.5, label='Equal Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/plots/time_ratio_dim{dim}.png')
        plt.close()

# ----------------------------------------
# Helper Functions for Visualization
# ----------------------------------------

from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_axes_locatable(ax):
    """
    Create a divider for an axes instance for adding subplots to its sides.
    This function replicates a key functionality from mpl_toolkits.axes_grid1.
    """
    # Simplified implementation for the purposes of this codebase
    divider = plt.figure(figsize=(10, 8))
    
    # Create new axes on the right and on the top of the current axes
    ax_histx = divider.add_subplot(3, 1, 1, sharex=ax)
    ax_histy = divider.add_subplot(1, 3, 3, sharey=ax)
    
    return divider

# ----------------------------------------
# Main Execution
# ----------------------------------------

def run_basic_example():
    """
    Run a basic example of the IMHK sampler on a 2D lattice.
    """
    print("Running basic 2D IMHK example...")
    
    # Parameters
    dim = 2
    sigma = 2.0
    num_samples = 2000
    burn_in = 1000
    
    # Identity basis
    B = matrix.identity(RR, dim)
    
    # Run IMHK sampler
    samples, acceptance_rate, all_samples, all_accepts = imhk_sampler(
        B, sigma, num_samples, burn_in=burn_in)
    
    print(f"Acceptance rate: {acceptance_rate:.4f}")
    
    # Plot trace
    plot_trace(samples, "basic_example_trace.png", "IMHK Sample Trace (2D)")
    
    # Compute autocorrelation
    acf_by_dim = compute_autocorrelation(samples)
    plot_autocorrelation(acf_by_dim, "basic_example_acf.png", "IMHK Autocorrelation (2D)")
    
    # Plot samples
    plot_2d_samples(samples, sigma, "basic_example_samples.png", B, "IMHK Samples (2D)")
    
    # Run Klein sampler for comparison
    klein_samples = [klein_sampler(B, sigma) for _ in range(num_samples)]
    plot_2d_samples(klein_samples, sigma, "basic_example_klein.png", B, "Klein Samples (2D)")
    
    # Compute statistical distances
    tv_distance_imhk = compute_total_variation_distance(samples, sigma, B)
    tv_distance_klein = compute_total_variation_distance(klein_samples, sigma, B)
    
    print(f"IMHK Total Variation distance: {tv_distance_imhk:.6f}")
    print(f"Klein Total Variation distance: {tv_distance_klein:.6f}")
    
    # Compute KL divergence
    kl_imhk = compute_kl_divergence(samples, sigma, B)
    kl_klein = compute_kl_divergence(klein_samples, sigma, B)
    
    print(f"IMHK KL divergence: {kl_imhk:.6f}")
    print(f"Klein KL divergence: {kl_klein:.6f}")
    
    print("Basic example completed.")

def run_comprehensive_tests():
    """
    Run comprehensive tests including parameter sweeps for the paper.
    """
    print("Running comprehensive parameter sweep...")
    
    # Dimensions to test
    dimensions = [2, 3, 4]
    
    # Sigma values to test
    # Include values below and above the smoothing parameter
    sigmas = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    # Basis types to test
    basis_types = ['identity', 'skewed', 'ill-conditioned']
    
    # Number of samples for each configuration
    num_samples = 2000
    
    # Run the parameter sweep
    results = parameter_sweep(
        dimensions=dimensions,
        sigmas=sigmas,
        basis_types=basis_types,
        num_samples=num_samples
    )
    
    # Run additional analysis
    compare_convergence_times(results)
    
    print("Comprehensive tests completed.")

def run_specific_experiment():
    """
    Run a specific experiment with detailed analysis for the paper.
    """
    print("Running specific experiment with detailed analysis...")
    
    # Parameters
    dim = 3
    sigma = 2.0
    num_samples = 5000
    burn_in = 2000
    basis_type = 'skewed'
    
    # Create a skewed basis
    B = matrix.identity(RR, dim)
    B[0, 1] = 1.5
    B[0, 2] = 0.5
    
    # Run the experiment
    result = run_experiment(
        dim=dim,
        sigma=sigma,
        num_samples=num_samples,
        basis_type=basis_type,
        compare_with_klein=True
    )
    
    print("Specific experiment completed.")

if __name__ == "__main__":
    # Create results directories if they don't exist
    os.makedirs('results/logs', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    set_random_seed(42)
    
    # Run basic example for quick testing
    run_basic_example()
    
    # Uncomment to run comprehensive tests
    # run_comprehensive_tests()
    
    # Uncomment to run a specific detailed experiment
    # run_specific_experiment()
    
    print("All experiments completed.")