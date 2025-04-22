from sage.structure.element import Vector
import numpy as np
from sage.all import *

def discrete_gaussian_sampler_1d(center, sigma):
    """
    Sample from a 1D discrete Gaussian distribution centered at 'center' with width 'sigma'.
    Uses an efficient table-based approach with rejection sampling for tails.
    
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

def imhk_sampler(B, sigma, num_samples, c=None, burn_in=1000):
    """
    Independent Metropolis-Hastings-Klein algorithm for sampling from a discrete Gaussian
    over a lattice.
    
    Args:
        B: The lattice basis matrix (rows are basis vectors)
        sigma: The standard deviation of the Gaussian
        num_samples: The number of samples to generate
        c: The center of the Gaussian (default: origin)
        burn_in: The number of initial samples to discard
        
    Returns:
        A list of lattice points sampled according to the discrete Gaussian,
        along with acceptance rate and other diagnostics
    """
    from .utils import discrete_gaussian_pdf

    n = B.nrows()
    if c is None:
        c = vector(RR, [0] * n)
    else:
        c = vector(RR, c)
    
    # Initialize the chain with a sample from Klein's algorithm
    current_sample = klein_sampler(B, sigma, c)
    current_density = discrete_gaussian_pdf(current_sample, sigma, c)
    
    samples = []
    acceptance_count = 0
    total_count = 0
    
    # Monitor individual samples for diagnostics
    all_samples = []  # Store all samples including burn-in for diagnostics
    all_accepts = []  # Store whether each proposal was accepted
    
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
        all_accepts.append(accept)
        all_samples.append(current_sample)
        
        # Store the sample if we're past the burn-in period
        if i >= burn_in:
            samples.append(current_sample)
    
    acceptance_rate = acceptance_count / total_count
    
    return samples, acceptance_rate, all_samples, all_accepts