#!/usr/bin/env python3
"""
Simple test to run the TV distance comparison with minimal configuration.
"""

import sys
sys.path.insert(0, '/Users/nicholaszhao/Documents/PhD/GitHub/quantum-mcmc-simulations/imhk_sampler')

from sage.all import *
from experiments import compare_tv_distance_vs_sigma
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Run a minimal TV distance comparison."""
    # Minimal parameters
    dimensions = [4]  # Just one dimension
    basis_types = ['identity']  # Just identity basis
    sigma_eta_ratios = [1.0]  # Just one ratio
    num_samples = 100  # Small sample size
    
    print("Running minimal TV distance comparison...")
    
    results = compare_tv_distance_vs_sigma(
        dimensions=dimensions,
        basis_types=basis_types,
        sigma_eta_ratios=sigma_eta_ratios,
        num_samples=num_samples,
        plot_results=False,  # Disable plots
        output_dir='results/test_tv_minimal'
    )
    
    print(f"\nGenerated {len(results)} results")
    
    for key, value in results.items():
        print(f"\nConfiguration: {key}")
        if 'error' in value:
            print(f"  ERROR: {value['error']}")
        else:
            print(f"  IMHK TV: {value.get('imhk_tv_distance', 'N/A')}")
            print(f"  Klein TV: {value.get('klein_tv_distance', 'N/A')}")
            print(f"  TV Ratio: {value.get('tv_ratio', 'N/A')}")
            print(f"  Acceptance: {value.get('imhk_acceptance_rate', 'N/A')}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()