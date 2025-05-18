#!/usr/bin/env python3
"""
Test a single experiment configuration to verify everything works.
"""

import sys
sys.path.insert(0, '/Users/nicholaszhao/Documents/PhD/GitHub/quantum-mcmc-simulations/imhk_sampler')

from sage.all import *
import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_single_config():
    """Test a single configuration."""
    # Direct imports
    from experiments import create_lattice_basis
    from samplers import imhk_sampler, klein_sampler
    from stats import compute_total_variation_distance
    
    # Parameters
    dim = 4
    sigma = 1.0
    num_samples = 100
    basis_type = 'identity'
    
    logger.info(f"Testing configuration: dim={dim}, sigma={sigma}, basis={basis_type}")
    
    # Create lattice basis
    B = create_lattice_basis(dim, basis_type)
    center = vector(RDF, [0] * dim)
    
    # Run IMHK sampler
    try:
        imhk_samples, acceptance_rate, _, _ = imhk_sampler(
            B, sigma, num_samples, center, burn_in=50)
        logger.info(f"IMHK sampling successful: acceptance rate = {acceptance_rate:.4f}")
        
        # Compute TV distance
        imhk_tv = compute_total_variation_distance(imhk_samples, sigma, B, center)
        logger.info(f"IMHK TV distance: {imhk_tv:.6f}")
    except Exception as e:
        logger.error(f"IMHK error: {e}")
        return False
    
    # Run Klein sampler
    try:
        klein_samples = [klein_sampler(B, sigma, center) for _ in range(num_samples)]
        logger.info("Klein sampling successful")
        
        # Compute TV distance
        klein_tv = compute_total_variation_distance(klein_samples, sigma, B, center)
        logger.info(f"Klein TV distance: {klein_tv:.6f}")
        
        # Compare
        if klein_tv > 0:
            tv_ratio = imhk_tv / klein_tv
            logger.info(f"TV ratio (IMHK/Klein): {tv_ratio:.4f}")
    except Exception as e:
        logger.error(f"Klein error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_single_config()
    if success:
        logger.info("Test completed successfully!")
    else:
        logger.error("Test failed!")
    sys.exit(0 if success else 1)