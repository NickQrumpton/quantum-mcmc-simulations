#!/usr/bin/env python3
"""
Test TV distance comparison without any visualization dependencies.
"""

import sys
sys.path.insert(0, '/Users/nicholaszhao/Documents/PhD/GitHub/quantum-mcmc-simulations/imhk_sampler')

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

def run_single_tv_experiment(dim, sigma, num_samples, basis_type):
    """Run a single TV experiment without visualization."""
    from experiments import create_lattice_basis
    from samplers import imhk_sampler, klein_sampler
    from stats import compute_total_variation_distance
    
    logger.info(f"Running: dim={dim}, sigma={float(sigma):.4f}, basis={basis_type}")
    
    # Create lattice basis and center
    B = create_lattice_basis(dim, basis_type)
    center = vector(RDF, [0] * dim)
    
    # Run IMHK sampler
    try:
        burn_in = 50
        start_time = time.time()
        imhk_samples, acceptance_rate, _, _ = imhk_sampler(
            B, sigma, num_samples, center, burn_in=burn_in)
        imhk_time = time.time() - start_time
        
        # Compute TV distance
        imhk_tv = compute_total_variation_distance(imhk_samples, sigma, B, center)
        
        logger.info(f"  IMHK: acceptance={acceptance_rate:.4f}, TV={imhk_tv:.6f}, time={imhk_time:.2f}s")
    except Exception as e:
        logger.error(f"  IMHK failed: {e}")
        return None
    
    # Run Klein sampler
    try:
        start_time = time.time()
        klein_samples = [klein_sampler(B, sigma, center) for _ in range(num_samples)]
        klein_time = time.time() - start_time
        
        # Compute TV distance
        klein_tv = compute_total_variation_distance(klein_samples, sigma, B, center)
        
        logger.info(f"  Klein: TV={klein_tv:.6f}, time={klein_time:.2f}s")
    except Exception as e:
        logger.error(f"  Klein failed: {e}")
        klein_tv = None
        klein_time = None
    
    # Prepare results
    results = {
        'dimension': dim,
        'sigma': float(sigma),
        'basis_type': basis_type,
        'num_samples': num_samples,
        'imhk_acceptance_rate': float(acceptance_rate),
        'imhk_tv_distance': float(imhk_tv),
        'imhk_time': float(imhk_time),
        'klein_tv_distance': float(klein_tv) if klein_tv is not None else None,
        'klein_time': float(klein_time) if klein_time is not None else None
    }
    
    if klein_tv is not None and klein_tv > 0:
        results['tv_ratio'] = float(imhk_tv / klein_tv)
        logger.info(f"  TV ratio (IMHK/Klein): {results['tv_ratio']:.4f}")
    
    return results

def main():
    """Run TV distance comparison without visualization."""
    from utils import calculate_smoothing_parameter
    
    # Test parameters
    dimensions = [4, 8]
    basis_types = ['identity', 'skewed']
    sigma_eta_ratios = [0.5, 1.0, 2.0]
    num_samples = 100
    
    output_dir = Path("results/tv_comparison_no_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for dim in dimensions:
        epsilon = 2**(-dim)
        eta = calculate_smoothing_parameter(dim, epsilon)
        logger.info(f"\nDimension {dim}, η={float(eta):.4f}")
        
        for basis_type in basis_types:
            for ratio in sigma_eta_ratios:
                sigma = ratio * eta
                
                result = run_single_tv_experiment(dim, sigma, num_samples, basis_type)
                
                if result:
                    result['sigma_eta_ratio'] = ratio
                    result['eta'] = float(eta)
                    all_results.append(result)
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create simple summary
    with open(output_dir / "summary.txt", 'w') as f:
        f.write("TV Distance Comparison Results\n")
        f.write("=" * 30 + "\n\n")
        
        f.write(f"Total experiments: {len(all_results)}\n\n")
        
        successful = [r for r in all_results if 'tv_ratio' in r]
        if successful:
            f.write("Successful comparisons:\n")
            for r in successful:
                f.write(f"Dim={r['dimension']}, Basis={r['basis_type']}, σ/η={r['sigma_eta_ratio']}: ")
                f.write(f"TV ratio={r['tv_ratio']:.4f}\n")
            
            avg_ratio = np.mean([r['tv_ratio'] for r in successful])
            f.write(f"\nAverage TV ratio: {avg_ratio:.4f}\n")
    
    logger.info(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()