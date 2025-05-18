#!/usr/bin/env python3
"""
Generate publication-quality TV distance comparison results between IMHK and Klein samplers.
This is the final production version that should work correctly.
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
        logging.FileHandler('publication_results_final.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("publication_results")

def run_single_experiment(dim, sigma, num_samples, basis_type):
    """Run a single experiment and return results."""
    from experiments import create_lattice_basis
    from samplers import imhk_sampler, klein_sampler
    from stats import compute_total_variation_distance
    from diagnostics import compute_ess
    
    logger.info(f"Running: dim={dim}, sigma={float(sigma):.4f}, basis={basis_type}")
    
    # Create lattice basis and center
    B = create_lattice_basis(dim, basis_type)
    center = vector(RDF, [0] * dim)
    
    # Run IMHK sampler
    try:
        burn_in = min(1000, num_samples // 2)
        start_time = time.time()
        imhk_samples, acceptance_rate, _, _ = imhk_sampler(
            B, sigma, num_samples, center, burn_in=burn_in)
        imhk_time = time.time() - start_time
        
        # Compute TV distance and ESS
        imhk_tv = compute_total_variation_distance(imhk_samples, sigma, B, center)
        imhk_ess = compute_ess(imhk_samples)
        
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
        'imhk_ess': [float(ess) for ess in imhk_ess],
        'klein_tv_distance': float(klein_tv) if klein_tv is not None else None,
        'klein_time': float(klein_time) if klein_time is not None else None
    }
    
    if klein_tv is not None and klein_tv > 0:
        results['tv_ratio'] = float(imhk_tv / klein_tv)
        logger.info(f"  TV ratio (IMHK/Klein): {results['tv_ratio']:.4f}")
    
    return results

def main():
    """Run comprehensive publication-quality experiments."""
    from utils import calculate_smoothing_parameter
    
    # Experiment parameters for publication
    dimensions = [8, 16, 32, 64]
    basis_types = ['identity', 'skewed', 'ill-conditioned']
    sigma_eta_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    num_samples = 5000
    
    # Create output directory
    output_dir = Path("results/publication/tv_distance_comparison_final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting publication-quality TV distance comparison")
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
        logger.info(f"Smoothing parameter η_{epsilon:.2e}(Λ) = {float(eta):.4f}")
        
        for basis_type in basis_types:
            logger.info(f"\nBasis type: {basis_type}")
            
            for ratio in sigma_eta_ratios:
                sigma = ratio * eta
                config_key = f"dim{dim}_{basis_type}_ratio{ratio}"
                
                result = run_single_experiment(dim, sigma, num_samples, basis_type)
                
                if result:
                    result['sigma_eta_ratio'] = ratio
                    result['eta'] = float(eta)
                    result['config_key'] = config_key
                    all_results.append(result)
    
    elapsed_time = time.time() - start_time
    logger.info(f"\nAll experiments completed in {elapsed_time/3600:.2f} hours")
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    with open(output_dir / "results.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    # Generate summary
    generate_summary(all_results, output_dir)
    
    logger.info(f"Results saved to {output_dir}")

def generate_summary(results, output_dir):
    """Generate a summary report of the results."""
    
    # Group results by configuration
    summary = {
        'total_experiments': len(results),
        'dimensions': sorted(list(set(r['dimension'] for r in results))),
        'basis_types': sorted(list(set(r['basis_type'] for r in results))),
        'sigma_eta_ratios': sorted(list(set(r['sigma_eta_ratio'] for r in results))),
        'best_improvements': [],
        'configuration_summaries': {}
    }
    
    # Find best improvements (lowest TV ratio)
    results_with_ratio = [r for r in results if 'tv_ratio' in r and r['tv_ratio'] is not None]
    results_with_ratio.sort(key=lambda x: x['tv_ratio'])
    summary['best_improvements'] = results_with_ratio[:10]
    
    # Summarize by configuration
    for dim in summary['dimensions']:
        for basis in summary['basis_types']:
            config_key = f"dim{dim}_{basis}"
            config_results = [r for r in results if r['dimension'] == dim and r['basis_type'] == basis]
            
            if config_results:
                tv_ratios = [r['tv_ratio'] for r in config_results if 'tv_ratio' in r and r['tv_ratio'] is not None]
                if tv_ratios:
                    summary['configuration_summaries'][config_key] = {
                        'dimension': dim,
                        'basis_type': basis,
                        'num_experiments': len(config_results),
                        'avg_tv_ratio': np.mean(tv_ratios),
                        'min_tv_ratio': min(tv_ratios),
                        'max_tv_ratio': max(tv_ratios),
                        'avg_acceptance_rate': np.mean([r['imhk_acceptance_rate'] for r in config_results])
                    }
    
    # Save summary
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate text report
    with open(output_dir / "report.txt", 'w') as f:
        f.write("PUBLICATION-QUALITY TV DISTANCE COMPARISON RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total experiments: {summary['total_experiments']}\n")
        f.write(f"Dimensions tested: {summary['dimensions']}\n")
        f.write(f"Basis types tested: {summary['basis_types']}\n")
        f.write(f"Sigma/eta ratios tested: {summary['sigma_eta_ratios']}\n\n")
        
        f.write("TOP 10 IMPROVEMENTS (IMHK over Klein):\n")
        f.write("-" * 40 + "\n")
        for i, result in enumerate(summary['best_improvements'][:10], 1):
            f.write(f"{i}. Dim={result['dimension']}, "
                   f"Basis={result['basis_type']}, "
                   f"σ/η={result['sigma_eta_ratio']}, "
                   f"TV Ratio={result['tv_ratio']:.6f}\n")
        
        f.write("\nCONFIGURATION SUMMARIES:\n")
        f.write("-" * 40 + "\n")
        for config_key, config in sorted(summary['configuration_summaries'].items()):
            f.write(f"\n{config_key}:\n")
            f.write(f"  Experiments run: {config['num_experiments']}\n")
            f.write(f"  Average TV ratio: {config['avg_tv_ratio']:.6f}\n")
            f.write(f"  Best TV ratio: {config['min_tv_ratio']:.6f}\n")
            f.write(f"  Worst TV ratio: {config['max_tv_ratio']:.6f}\n")
            f.write(f"  Average acceptance rate: {config['avg_acceptance_rate']:.4f}\n")
        
        f.write("\nCONCLUSIONS:\n")
        f.write("-" * 40 + "\n")
        
        # Overall best improvement
        if summary['best_improvements']:
            best = summary['best_improvements'][0]
            improvement = (1 - best['tv_ratio']) * 100
            f.write(f"Best improvement: {improvement:.1f}% "
                   f"(Dim={best['dimension']}, "
                   f"Basis={best['basis_type']}, "
                   f"σ/η={best['sigma_eta_ratio']})\n\n")
        
        # Average improvements by basis type
        for basis in summary['basis_types']:
            basis_configs = [v for k, v in summary['configuration_summaries'].items() 
                           if v['basis_type'] == basis]
            if basis_configs:
                avg_ratio = np.mean([c['avg_tv_ratio'] for c in basis_configs])
                avg_improvement = (1 - avg_ratio) * 100
                f.write(f"Average improvement for {basis} basis: {avg_improvement:.1f}%\n")
    
    logger.info("Summary and report generated")

if __name__ == "__main__":
    main()