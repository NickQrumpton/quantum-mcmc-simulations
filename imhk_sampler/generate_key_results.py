#!/usr/bin/env python3
"""
Generate key results demonstrating the compare_tv_distance_vs_sigma function.
Uses smaller parameters for faster execution while still showing meaningful results.
"""

import sys
sys.path.insert(0, '/Users/nicholaszhao/Documents/PhD/GitHub/quantum-mcmc-simulations/imhk_sampler')

from sage.all import *
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("key_results")

def run_key_experiments():
    """Run key experiments using compare_tv_distance_vs_sigma."""
    from experiments import compare_tv_distance_vs_sigma
    
    # Key parameters for demonstration
    dimensions = [4, 8]  # Smaller set
    basis_types = ['identity', 'skewed']  # Core basis types
    sigma_eta_ratios = [0.5, 1.0, 2.0]  # Key ratios
    num_samples = 200  # Smaller sample size
    
    logger.info("Running key experiments with compare_tv_distance_vs_sigma")
    logger.info(f"Dimensions: {dimensions}")
    logger.info(f"Basis types: {basis_types}")
    logger.info(f"Sigma/eta ratios: {sigma_eta_ratios}")
    logger.info(f"Number of samples: {num_samples}")
    
    # Run the comparison
    results = compare_tv_distance_vs_sigma(
        dimensions=dimensions,
        basis_types=basis_types,
        sigma_eta_ratios=sigma_eta_ratios,
        num_samples=num_samples,
        plot_results=True,  # Enable plots
        output_dir='results/key_demonstration'
    )
    
    return results

def create_summary_report(results, output_dir):
    """Create a concise summary report."""
    with open(output_dir / "summary.txt", 'w') as f:
        f.write("KEY RESULTS: IMHK vs Klein TV Distance Comparison\n")
        f.write("=" * 50 + "\n\n")
        
        # Count successful experiments
        successful = [r for r in results.values() if 'tv_ratio' in r and r['tv_ratio'] is not None]
        failed = [r for r in results.values() if 'error' in r]
        
        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n\n")
        
        if successful:
            f.write("SUCCESSFUL EXPERIMENTS:\n")
            f.write("-" * 30 + "\n")
            
            for result in successful:
                f.write(f"Dim={result['dimension']}, ")
                f.write(f"Basis={result['basis_type']}, ")
                f.write(f"σ/η={result['sigma_eta_ratio']}\n")
                f.write(f"  IMHK TV: {result['imhk_tv_distance']:.6f}\n")
                f.write(f"  Klein TV: {result['klein_tv_distance']:.6f}\n")
                f.write(f"  TV Ratio: {result['tv_ratio']:.4f}\n")
                f.write(f"  Acceptance: {result['imhk_acceptance_rate']:.4f}\n\n")
            
            # Summary statistics
            tv_ratios = [r['tv_ratio'] for r in successful]
            acceptance_rates = [r['imhk_acceptance_rate'] for r in successful]
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average TV ratio: {np.mean(tv_ratios):.4f}\n")
            f.write(f"Best TV ratio: {min(tv_ratios):.4f}\n")
            f.write(f"Average acceptance rate: {np.mean(acceptance_rates):.4f}\n")
        
        if failed:
            f.write("\nFAILED EXPERIMENTS:\n")
            f.write("-" * 30 + "\n")
            for result in failed:
                f.write(f"Dim={result['dimension']}, ")
                f.write(f"Basis={result['basis_type']}, ")
                f.write(f"σ/η={result['sigma_eta_ratio']}\n")
                f.write(f"  Error: {result['error']}\n\n")

def main():
    """Main function to generate key results."""
    output_dir = Path("results/key_demonstration")
    
    # Run experiments
    results = run_key_experiments()
    
    # Create summary
    create_summary_report(results, output_dir)
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info("Check summary.txt for key findings")
    logger.info("Check plots/ for visualizations")

if __name__ == "__main__":
    main()