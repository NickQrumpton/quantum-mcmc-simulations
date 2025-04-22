from sage.structure.element import Vector
from sage.all import *

def discrete_gaussian_pdf(x, sigma, center=None):
    """
    Compute the probability density function of a discrete Gaussian distribution.
    
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
    
    if isinstance(x, Vector):
        # Compute the squared norm of (x - center)
        squared_norm = sum((xi - ci) ** 2 for xi, ci in zip(x, center))
        return exp(-squared_norm / (2 * sigma ** 2))
    else:
        # Scalar case
        return exp(-(x - center) ** 2 / (2 * sigma ** 2))

def precompute_discrete_gaussian_probabilities(sigma, center=0, radius=6):
    """
    Precompute discrete Gaussian probabilities for integers within radius*sigma of center.
    
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