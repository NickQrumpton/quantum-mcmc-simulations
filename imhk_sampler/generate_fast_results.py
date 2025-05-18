#!/usr/bin/env python3
"""
Generate publication results with faster TV distance estimation.
Uses a subset of lattice points for TV distance calculation.
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fast_results")

def fast_tv_distance(samples, sigma, lattice_basis, center):
    """
    Fast approximation of TV distance using histogram-based approach.
    """
    from utils import discrete_gaussian_pdf
    
    # Convert samples to numpy array
    samples_np = np.array([[float(x) for x in sample] for sample in samples])
    
    # Create a grid around the center with radius proportional to sigma
    dim = len(samples[0])
    grid_size = int(6 * sigma)  # Grid extends to ~6 sigma
    n_bins = 20  # Number of bins per dimension (reduced for speed)
    
    # Create histogram
    ranges = [(float(center[i] - grid_size), float(center[i] + grid_size)) for i in range(dim)]
    hist, edges = np.histogramdd(samples_np, bins=n_bins, range=ranges)
    
    # Normalize histogram
    hist = hist / len(samples)
    
    # Calculate theoretical probabilities on the same grid
    theoretical_sum = 0
    tv_distance = 0
    
    # Sample points from the grid for theoretical calculation
    n_sample_points = min(1000, n_bins**dim)  # Limit number of points
    
    for _ in range(n_sample_points):
        # Random point in grid
        idx = tuple(np.random.randint(0, n_bins) for _ in range(dim))
        if hist[idx] > 0:
            # Get center of bin
            point = vector(RDF, [
                (edges[i][idx[i]] + edges[i][idx[i]+1]) / 2
                for i in range(dim)
            ])
            
            # Calculate theoretical probability
            theoretical_prob = discrete_gaussian_pdf(point, sigma, center)
            bin_volume = np.prod([(edges[i][1] - edges[i][0]) for i in range(dim)])
            theoretical_prob *= bin_volume
            
            theoretical_sum += theoretical_prob
            tv_distance += abs(hist[idx] - theoretical_prob)
    
    # Rough normalization
    if theoretical_sum > 0:
        tv_distance = tv_distance / (2 * theoretical_sum)
    
    return min(tv_distance, 1.0)  # Cap at 1.0

def run_fast_experiment(dim, sigma, num_samples, basis_type):
    """Run experiment with fast TV distance calculation."""
    from experiments import create_lattice_basis
    from samplers import imhk_sampler, klein_sampler
    
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
        
        # Fast TV distance
        imhk_tv = fast_tv_distance(imhk_samples, sigma, B, center)
        
        logger.info(f"  IMHK: acceptance={acceptance_rate:.4f}, TV={imhk_tv:.6f}, time={imhk_time:.2f}s")
    except Exception as e:
        logger.error(f"  IMHK failed: {e}")
        return None
    
    # Run Klein sampler
    try:
        start_time = time.time()
        klein_samples = [klein_sampler(B, sigma, center) for _ in range(num_samples)]
        klein_time = time.time() - start_time
        
        # Fast TV distance
        klein_tv = fast_tv_distance(klein_samples, sigma, B, center)
        
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
    """Run experiments with fast TV distance estimation."""
    from utils import calculate_smoothing_parameter
    
    # Experiment parameters
    dimensions = [4, 8, 16, 32]
    basis_types = ['identity', 'skewed', 'ill-conditioned']
    sigma_eta_ratios = [0.5, 1.0, 2.0, 4.0]
    num_samples = 1000
    
    # Create output directory
    output_dir = Path("results/publication/fast_tv_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting fast TV distance comparison")
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
                
                result = run_fast_experiment(dim, sigma, num_samples, basis_type)
                
                if result:
                    result['sigma_eta_ratio'] = ratio
                    result['eta'] = float(eta)
                    all_results.append(result)
    
    elapsed_time = time.time() - start_time
    logger.info(f"\nAll experiments completed in {elapsed_time/60:.2f} minutes")
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary
    generate_summary(all_results, output_dir)
    
    logger.info(f"Results saved to {output_dir}")

def generate_summary(results, output_dir):
    """Generate summary of results."""
    
    with open(output_dir / "summary.txt", 'w') as f:
        f.write("FAST TV DISTANCE COMPARISON RESULTS\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total experiments: {len(results)}\n")
        
        # Best performances
        sorted_results = sorted([r for r in results if 'tv_ratio' in r], 
                              key=lambda x: x['tv_ratio'])
        
        f.write("\nBEST PERFORMANCES (lowest TV ratio):\n")
        for i, result in enumerate(sorted_results[:10], 1):
            f.write(f"{i}. Dim={result['dimension']}, "
                   f"Basis={result['basis_type']}, "
                   f"σ/η={result['sigma_eta_ratio']}: "
                   f"TV ratio={result['tv_ratio']:.4f}\n")
        
        # Summary by dimension
        f.write("\nAVERAGE TV RATIO BY DIMENSION:\n")
        for dim in [4, 8, 16, 32]:
            dim_results = [r for r in results if r['dimension'] == dim and 'tv_ratio' in r]
            if dim_results:
                avg_ratio = np.mean([r['tv_ratio'] for r in dim_results])
                f.write(f"Dimension {dim}: {avg_ratio:.4f}\n")
        
        # Summary by basis type
        f.write("\nAVERAGE TV RATIO BY BASIS TYPE:\n")
        for basis in ['identity', 'skewed', 'ill-conditioned']:
            basis_results = [r for r in results if r['basis_type'] == basis and 'tv_ratio' in r]
            if basis_results:
                avg_ratio = np.mean([r['tv_ratio'] for r in basis_results])
                avg_improvement = (1 - avg_ratio) * 100
                f.write(f"{basis}: {avg_ratio:.4f} ({avg_improvement:.1f}% improvement)\n")

if __name__ == "__main__":
    main()