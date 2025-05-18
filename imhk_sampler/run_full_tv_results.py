#!/usr/bin/env sage
"""
Full publication-quality TV distance comparison run for IMHK sampler experiments.

This script runs the complete set of experiments for comparing IMHK and Klein samplers
across multiple dimensions, basis types, and sigma values for publication-quality results.
"""

import os
import sys
import json
import time
import logging
from sage.all import *

# Add the imhk_sampler directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the experiments module
from experiments import compare_tv_distance_vs_sigma, init_directories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_tv_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_full_tv_comparison():
    """Run the full TV distance comparison with publication-quality parameters."""
    logger.info("Starting full TV distance comparison for publication")
    
    # Initialize directories
    init_directories("results/publication_tv")
    
    # Set up parameters for publication-quality results
    dimensions = [2, 4, 8, 16, 32, 64]
    basis_types = ['identity', 'skewed', 'ill-conditioned']
    sigma_eta_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
    num_samples = 10000
    
    logger.info(f"Configuration:")
    logger.info(f"  Dimensions: {dimensions}")
    logger.info(f"  Basis types: {basis_types}")
    logger.info(f"  Sigma/eta ratios: {sigma_eta_ratios}")
    logger.info(f"  Samples per experiment: {num_samples}")
    
    # Estimate runtime
    total_experiments = len(dimensions) * len(basis_types) * len(sigma_eta_ratios)
    logger.info(f"Total experiments to run: {total_experiments}")
    logger.info(f"Estimated runtime: {total_experiments * 2} - {total_experiments * 5} minutes")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Run the comparison
        results = compare_tv_distance_vs_sigma(
            dimensions=dimensions,
            basis_types=basis_types,
            sigma_eta_ratios=sigma_eta_ratios,
            num_samples=num_samples,
            plot_results=True,
            output_dir="results/publication_tv"
        )
        
        # Calculate runtime
        runtime = time.time() - start_time
        logger.info(f"Total runtime: {runtime:.2f} seconds ({runtime/60:.1f} minutes)")
        
        # Create summary report
        summary = {
            'dimensions': dimensions,
            'basis_types': basis_types,
            'sigma_eta_ratios': sigma_eta_ratios,
            'num_samples': num_samples,
            'total_experiments': total_experiments,
            'runtime_seconds': runtime,
            'runtime_minutes': runtime/60,
            'successful_experiments': 0,
            'failed_experiments': 0,
            'key_findings': []
        }
        
        # Analyze results
        for key, result in results.items():
            if 'error' in result:
                summary['failed_experiments'] += 1
            else:
                summary['successful_experiments'] += 1
                
                # Check for significant findings
                if result.get('tv_ratio') and result['tv_ratio'] < 0.9:
                    summary['key_findings'].append({
                        'dimension': result['dimension'],
                        'basis_type': result['basis_type'],
                        'sigma_eta_ratio': result['sigma_eta_ratio'],
                        'tv_ratio': result['tv_ratio'],
                        'finding': 'IMHK shows improvement over Klein'
                    })
        
        # Save summary
        with open("results/publication_tv/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Successful experiments: {summary['successful_experiments']}")
        logger.info(f"Failed experiments: {summary['failed_experiments']}")
        logger.info(f"Key findings: {len(summary['key_findings'])}")
        
        # Print key findings
        if summary['key_findings']:
            logger.info("\nKey Findings:")
            for finding in summary['key_findings'][:5]:  # Show top 5
                logger.info(f"  Dim {finding['dimension']}, {finding['basis_type']}, "
                          f"σ/η={finding['sigma_eta_ratio']}: "
                          f"TV ratio={finding['tv_ratio']:.4f}")
        
        logger.info("\nFull TV distance comparison completed successfully!")
        logger.info(f"Results saved to: results/publication_tv/")
        
    except Exception as e:
        logger.error(f"Error during TV distance comparison: {e}")
        raise
    
    return results

if __name__ == "__main__":
    logger.info("Starting IMHK sampler full TV distance comparison")
    
    # Run the comparison
    results = run_full_tv_comparison()
    
    # Print completion message
    print("\n" + "="*60)
    print("IMHK Sampler TV Distance Comparison Completed!")
    print("="*60)
    print("\nResults saved to: results/publication_tv/")
    print("Check the following files:")
    print("  - tv_distance_comparison.json: Numerical results")
    print("  - tv_distance_comparison.pkl: Full results with data")
    print("  - summary.json: Experiment summary and key findings")
    print("  - plots/: Visualization plots")
    print("  - full_tv_results.log: Detailed execution log")
    print("\n" + "="*60)