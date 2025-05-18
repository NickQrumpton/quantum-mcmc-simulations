#!/usr/bin/env python3
"""
Run a minimal experiment to generate publication results without visualization issues.
This directly runs the sampling algorithms to get TV distance measurements.
"""

from sage.all import *
import numpy as np
import time
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_minimal_experiment(dim=4, sigma=1.0, num_samples=100, basis_type='identity'):
    """Run a minimal experiment without plots."""
    from utils import calculate_smoothing_parameter
    from experiments import create_lattice_basis
    from samplers import imhk_sampler, klein_sampler
    from stats import compute_total_variation_distance
    
    logger.info(f"Running experiment: dim={dim}, sigma={sigma}, basis={basis_type}")
    
    # Create lattice basis
    B = create_lattice_basis(dim, basis_type)
    center = vector(RR, [0] * dim)
    
    # Run IMHK sampler
    burn_in = min(100, num_samples)
    try:
        imhk_samples, acceptance_rate, all_samples, all_accepts = imhk_sampler(
            B, sigma, num_samples, center, burn_in=burn_in)
        imhk_tv = compute_total_variation_distance(imhk_samples, sigma, B, center)
        logger.info(f"IMHK: acceptance={acceptance_rate:.4f}, TV={imhk_tv:.6f}")
    except Exception as e:
        logger.error(f"IMHK failed: {e}")
        return None
    
    # Run Klein sampler
    try:
        klein_samples = [klein_sampler(B, sigma, center) for _ in range(num_samples)]
        klein_tv = compute_total_variation_distance(klein_samples, sigma, B, center)
        logger.info(f"Klein: TV={klein_tv:.6f}")
    except Exception as e:
        logger.error(f"Klein failed: {e}")
        klein_tv = None
    
    # Return results
    results = {
        'dimension': dim,
        'sigma': sigma,
        'basis_type': basis_type,
        'num_samples': num_samples,
        'imhk_acceptance_rate': float(acceptance_rate),
        'imhk_tv_distance': float(imhk_tv),
        'klein_tv_distance': float(klein_tv) if klein_tv is not None else None,
        'tv_ratio': float(imhk_tv/klein_tv) if klein_tv is not None and klein_tv > 0 else None
    }
    
    return results

def main():
    """Run a set of minimal experiments."""
    output_dir = Path("results/minimal_tv_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import at the top level
    from utils import calculate_smoothing_parameter
    
    # Test parameters
    dimensions = [4, 8]
    basis_types = ['identity', 'skewed']
    sigma_values = [0.5, 1.0, 2.0]
    num_samples = 200
    
    all_results = []
    
    for dim in dimensions:
        # Calculate eta for this dimension
        epsilon = 2**(-dim)
        eta = calculate_smoothing_parameter(dim, epsilon)
        
        for basis_type in basis_types:
            for sigma_ratio in sigma_values:
                sigma = sigma_ratio * eta
                
                logger.info(f"\nRunning: dim={dim}, basis={basis_type}, sigma/eta={sigma_ratio}")
                result = run_minimal_experiment(dim, sigma, num_samples, basis_type)
                
                if result:
                    result['sigma_eta_ratio'] = sigma_ratio
                    result['eta'] = float(eta)
                    all_results.append(result)
                    logger.info(f"Result: TV ratio = {result.get('tv_ratio', 'N/A')}")
    
    # Save results
    with open(output_dir / "minimal_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary
    summary = "MINIMAL EXPERIMENT SUMMARY\n" + "="*30 + "\n\n"
    for result in all_results:
        summary += f"Dim={result['dimension']}, "
        summary += f"Basis={result['basis_type']}, "
        summary += f"σ/η={result['sigma_eta_ratio']}: "
        if result['tv_ratio'] is not None:
            summary += f"TV ratio = {result['tv_ratio']:.4f}\n"
        else:
            summary += "TV ratio = N/A\n"
    
    with open(output_dir / "summary.txt", 'w') as f:
        f.write(summary)
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info("Done!")

if __name__ == "__main__":
    main()