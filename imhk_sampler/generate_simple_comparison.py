#!/usr/bin/env python3
"""
Generate simple comparison results between IMHK and Klein samplers.
Focuses on acceptance rates and basic statistics rather than TV distance.
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
logger = logging.getLogger("simple_comparison")

def sample_variance(samples):
    """Calculate sample variance for comparison."""
    samples_np = np.array([[float(x) for x in sample] for sample in samples])
    return np.mean(np.var(samples_np, axis=0))

def run_comparison(dim, sigma, num_samples, basis_type):
    """Run IMHK vs Klein comparison."""
    from experiments import create_lattice_basis
    from samplers import imhk_sampler, klein_sampler
    
    logger.info(f"Comparing: dim={dim}, sigma={float(sigma):.4f}, basis={basis_type}")
    
    # Create lattice basis and center
    B = create_lattice_basis(dim, basis_type)
    center = vector(RDF, [0] * dim)
    
    results = {
        'dimension': dim,
        'sigma': float(sigma),
        'basis_type': basis_type,
        'num_samples': num_samples
    }
    
    # Run IMHK sampler
    try:
        burn_in = 50
        start_time = time.time()
        imhk_samples, acceptance_rate, _, _ = imhk_sampler(
            B, sigma, num_samples, center, burn_in=burn_in)
        imhk_time = time.time() - start_time
        
        imhk_variance = sample_variance(imhk_samples)
        
        results['imhk_acceptance_rate'] = float(acceptance_rate)
        results['imhk_variance'] = float(imhk_variance)
        results['imhk_time'] = float(imhk_time)
        logger.info(f"  IMHK: acceptance={acceptance_rate:.4f}, variance={imhk_variance:.4f}, time={imhk_time:.2f}s")
    except Exception as e:
        logger.error(f"  IMHK failed: {e}")
        return None
    
    # Run Klein sampler
    try:
        start_time = time.time()
        klein_samples = [klein_sampler(B, sigma, center) for _ in range(num_samples)]
        klein_time = time.time() - start_time
        
        klein_variance = sample_variance(klein_samples)
        
        results['klein_variance'] = float(klein_variance)
        results['klein_time'] = float(klein_time)
        logger.info(f"  Klein: variance={klein_variance:.4f}, time={klein_time:.2f}s")
    except Exception as e:
        logger.error(f"  Klein failed: {e}")
        results['klein_variance'] = None
        results['klein_time'] = None
    
    # Calculate relative metrics
    if 'klein_variance' in results and results['klein_variance'] is not None:
        results['variance_ratio'] = results['imhk_variance'] / results['klein_variance']
        results['time_ratio'] = results['imhk_time'] / results['klein_time']
    
    return results

def main():
    """Run simple comparison experiments."""
    from utils import calculate_smoothing_parameter
    
    # Experiment parameters
    dimensions = [4, 8, 16, 32]
    basis_types = ['identity', 'skewed', 'ill-conditioned']
    sigma_eta_ratios = [0.5, 1.0, 2.0, 4.0]
    num_samples = 500
    
    # Create output directory
    output_dir = Path("results/publication/simple_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting simple comparison")
    all_results = []
    
    for dim in dimensions:
        # Calculate smoothing parameter
        epsilon = 2**(-dim)
        eta = calculate_smoothing_parameter(dim, epsilon)
        logger.info(f"\n=== Dimension {dim}, η = {float(eta):.4f} ===")
        
        for basis_type in basis_types:
            for ratio in sigma_eta_ratios:
                sigma = ratio * eta
                
                result = run_comparison(dim, sigma, num_samples, basis_type)
                
                if result:
                    result['sigma_eta_ratio'] = ratio
                    all_results.append(result)
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate report
    with open(output_dir / "report.txt", 'w') as f:
        f.write("SIMPLE COMPARISON RESULTS\n")
        f.write("========================\n\n")
        
        # Key findings
        f.write("KEY FINDINGS:\n\n")
        
        # Best acceptance rates
        f.write("Best IMHK Acceptance Rates:\n")
        sorted_results = sorted(all_results, key=lambda x: x['imhk_acceptance_rate'], reverse=True)
        for i, result in enumerate(sorted_results[:5], 1):
            f.write(f"{i}. Dim={result['dimension']}, "
                   f"Basis={result['basis_type']}, "
                   f"σ/η={result['sigma_eta_ratio']}: "
                   f"{result['imhk_acceptance_rate']:.4f}\n")
        
        f.write("\nAcceptance Rate by Basis Type:\n")
        for basis in basis_types:
            basis_results = [r for r in all_results if r['basis_type'] == basis]
            if basis_results:
                avg_acceptance = np.mean([r['imhk_acceptance_rate'] for r in basis_results])
                f.write(f"{basis}: {avg_acceptance:.4f}\n")
        
        f.write("\nAcceptance Rate by σ/η Ratio:\n")
        for ratio in sigma_eta_ratios:
            ratio_results = [r for r in all_results if r['sigma_eta_ratio'] == ratio]
            if ratio_results:
                avg_acceptance = np.mean([r['imhk_acceptance_rate'] for r in ratio_results])
                f.write(f"σ/η = {ratio}: {avg_acceptance:.4f}\n")
        
        # Variance comparison
        f.write("\nVariance Ratio (IMHK/Klein):\n")
        variance_results = [r for r in all_results if 'variance_ratio' in r]
        if variance_results:
            avg_var_ratio = np.mean([r['variance_ratio'] for r in variance_results])
            f.write(f"Average: {avg_var_ratio:.4f}\n")
            f.write(f"Best: {min(r['variance_ratio'] for r in variance_results):.4f}\n")
            f.write(f"Worst: {max(r['variance_ratio'] for r in variance_results):.4f}\n")
        
        # Time comparison
        f.write("\nTime Ratio (IMHK/Klein):\n")
        time_results = [r for r in all_results if 'time_ratio' in r]
        if time_results:
            avg_time_ratio = np.mean([r['time_ratio'] for r in time_results])
            f.write(f"Average: {avg_time_ratio:.4f}\n")
            f.write(f"Best: {min(r['time_ratio'] for r in time_results):.4f}\n")
            f.write(f"Worst: {max(r['time_ratio'] for r in time_results):.4f}\n")
    
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()