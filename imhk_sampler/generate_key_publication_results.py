#!/usr/bin/env python3
"""
Generate key publication-quality results with a focused set of experiments.
This version runs fewer but more strategic experiments to demonstrate the main findings.
"""

import sys
sys.path.insert(0, '/Users/nicholaszhao/Documents/PhD/GitHub/quantum-mcmc-simulations/imhk_sampler')

from sage.all import *
import numpy as np
import time
import json
import pickle
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('key_publication_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("publication_results")

def run_experiment(dim, sigma, num_samples, basis_type):
    """Run a single experiment and return results."""
    from experiments import create_lattice_basis
    from samplers import imhk_sampler, klein_sampler
    from stats import compute_total_variation_distance
    
    logger.info(f"Running: dim={dim}, sigma={float(sigma):.4f}, basis={basis_type}")
    
    # Create lattice basis and center
    B = create_lattice_basis(dim, basis_type)
    center = vector(RDF, [0] * dim)
    
    # Run IMHK sampler
    try:
        burn_in = 100  # Smaller burn-in for efficiency
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
    """Run key publication experiments."""
    from utils import calculate_smoothing_parameter
    
    # Focused experiment parameters - key configurations only
    dimensions = [4, 8, 16]  # Skip 32 and 64 for now
    basis_types = ['identity', 'skewed', 'ill-conditioned']
    sigma_eta_ratios = [0.5, 1.0, 2.0, 4.0]  # Key ratios only
    num_samples = 500  # Smaller sample size for efficiency
    
    # Create output directory
    output_dir = Path("results/publication/key_tv_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting key publication results generation")
    logger.info(f"Dimensions: {dimensions}")
    logger.info(f"Basis types: {basis_types}")
    logger.info(f"Sigma/eta ratios: {sigma_eta_ratios}")
    logger.info(f"Number of samples: {num_samples}")
    
    all_results = []
    start_time = time.time()
    
    for dim in dimensions:
        # Calculate smoothing parameter
        epsilon = 2**(-dim)
        eta = calculate_smoothing_parameter(dim, epsilon)
        logger.info(f"\n=== Dimension {dim} ===")
        logger.info(f"Smoothing parameter η = {float(eta):.4f}")
        
        for basis_type in basis_types:
            logger.info(f"\nBasis type: {basis_type}")
            
            for ratio in sigma_eta_ratios:
                sigma = ratio * eta
                
                result = run_experiment(dim, sigma, num_samples, basis_type)
                
                if result:
                    result['sigma_eta_ratio'] = ratio
                    result['eta'] = float(eta)
                    all_results.append(result)
    
    elapsed_time = time.time() - start_time
    logger.info(f"\nAll experiments completed in {elapsed_time/60:.2f} minutes")
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    with open(output_dir / "results.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    # Generate report
    generate_report(all_results, output_dir)
    
    logger.info(f"Results saved to {output_dir}")

def generate_report(results, output_dir):
    """Generate a concise report of key findings."""
    
    with open(output_dir / "key_findings.txt", 'w') as f:
        f.write("KEY PUBLICATION RESULTS: TV DISTANCE COMPARISON\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total experiments: {len(results)}\n\n")
        
        # Best performances (lowest TV ratio)
        sorted_results = sorted([r for r in results if 'tv_ratio' in r], 
                              key=lambda x: x['tv_ratio'])
        
        f.write("BEST PERFORMANCES (IMHK over Klein):\n")
        f.write("-" * 30 + "\n")
        for i, result in enumerate(sorted_results[:5], 1):
            improvement = (1 - result['tv_ratio']) * 100
            f.write(f"{i}. Dim={result['dimension']}, "
                   f"Basis={result['basis_type']}, "
                   f"σ/η={result['sigma_eta_ratio']}\n")
            f.write(f"   TV Ratio={result['tv_ratio']:.4f} "
                   f"({improvement:.1f}% improvement)\n\n")
        
        # Summary by dimension
        f.write("\nSUMMARY BY DIMENSION:\n")
        f.write("-" * 30 + "\n")
        for dim in [4, 8, 16]:
            dim_results = [r for r in results if r['dimension'] == dim and 'tv_ratio' in r]
            if dim_results:
                avg_ratio = np.mean([r['tv_ratio'] for r in dim_results])
                best_ratio = min([r['tv_ratio'] for r in dim_results])
                f.write(f"Dimension {dim}:\n")
                f.write(f"  Average TV ratio: {avg_ratio:.4f}\n")
                f.write(f"  Best TV ratio: {best_ratio:.4f}\n\n")
        
        # Summary by basis type
        f.write("\nSUMMARY BY BASIS TYPE:\n")
        f.write("-" * 30 + "\n")
        for basis in ['identity', 'skewed', 'ill-conditioned']:
            basis_results = [r for r in results if r['basis_type'] == basis and 'tv_ratio' in r]
            if basis_results:
                avg_ratio = np.mean([r['tv_ratio'] for r in basis_results])
                best_ratio = min([r['tv_ratio'] for r in basis_results])
                f.write(f"{basis.capitalize()} basis:\n")
                f.write(f"  Average TV ratio: {avg_ratio:.4f}\n")
                f.write(f"  Best TV ratio: {best_ratio:.4f}\n\n")
        
        # Main conclusions
        f.write("\nMAIN CONCLUSIONS:\n")
        f.write("-" * 30 + "\n")
        
        if sorted_results:
            best = sorted_results[0]
            improvement = (1 - best['tv_ratio']) * 100
            f.write(f"1. Best improvement: {improvement:.1f}% "
                   f"(Dim={best['dimension']}, {best['basis_type']}, σ/η={best['sigma_eta_ratio']})\n")
        
        # Check performance vs sigma/eta ratio
        low_ratio_results = [r for r in results if r['sigma_eta_ratio'] <= 1.0 and 'tv_ratio' in r]
        high_ratio_results = [r for r in results if r['sigma_eta_ratio'] > 1.0 and 'tv_ratio' in r]
        
        if low_ratio_results and high_ratio_results:
            avg_low = np.mean([r['tv_ratio'] for r in low_ratio_results])
            avg_high = np.mean([r['tv_ratio'] for r in high_ratio_results])
            f.write(f"2. IMHK performs better at {'low' if avg_low < avg_high else 'high'} σ/η ratios\n")
            f.write(f"   (avg TV ratio: low σ/η={avg_low:.4f}, high σ/η={avg_high:.4f})\n")
    
    logger.info("Report generated")

if __name__ == "__main__":
    main()