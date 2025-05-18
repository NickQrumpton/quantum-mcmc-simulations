#!/usr/bin/env python3
"""
Small test version of the publication results generation.
"""

import sys
sys.path.insert(0, '/Users/nicholaszhao/Documents/PhD/GitHub/quantum-mcmc-simulations/imhk_sampler')

from generate_publication_results_final import run_single_experiment
from utils import calculate_smoothing_parameter
from sage.all import *
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run a small test."""
    # Small test parameters
    dimensions = [4, 8]
    basis_types = ['identity', 'skewed']
    sigma_eta_ratios = [0.5, 1.0, 2.0]
    num_samples = 200
    
    logger.info("Running small test")
    
    for dim in dimensions:
        epsilon = 2**(-dim)
        eta = calculate_smoothing_parameter(dim, epsilon)
        logger.info(f"\nDimension {dim}, eta={float(eta):.4f}")
        
        for basis_type in basis_types:
            for ratio in sigma_eta_ratios:
                sigma = ratio * eta
                result = run_single_experiment(dim, sigma, num_samples, basis_type)
                
                if result and 'tv_ratio' in result:
                    logger.info(f"SUCCESS: dim={dim}, basis={basis_type}, σ/η={ratio}, TV ratio={result['tv_ratio']:.4f}")
                else:
                    logger.error(f"FAILED: dim={dim}, basis={basis_type}, σ/η={ratio}")
    
    logger.info("Test complete")

if __name__ == "__main__":
    main()