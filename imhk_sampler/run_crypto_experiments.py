#!/usr/bin/env sage -python
"""
Run comprehensive experiments with cryptographic lattice bases.
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import create_lattice_basis
from samplers import imhk_sampler_wrapper, klein_sampler_wrapper
from parameter_config import compute_smoothing_parameter
from stats import compute_total_variation_distance
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_single_experiment(basis_type, dim, sigma_ratio=2.0, num_samples=1000):
    """Run a single experiment configuration."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Running experiment: {basis_type} basis, dimension {dim}")
    
    results = {
        'basis_type': basis_type,
        'dimension': dim,
        'sigma_ratio': sigma_ratio,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Create the basis
        basis_info = create_lattice_basis(dim, basis_type)
        
        # Determine sigma based on basis type
        if isinstance(basis_info, tuple):
            # Structured lattice (NTRU, PrimeCyclotomic)
            poly_mod, q = basis_info
            degree = poly_mod.degree()
            # Use a reasonable sigma for cryptographic applications
            sigma = float(sqrt(q) / 10)  # Heuristic for cryptographic parameters
            logger.info(f"Structured lattice: degree={degree}, q={q}, sigma={sigma}")
            results['polynomial_degree'] = degree
            results['modulus'] = q
        else:
            # Matrix-based lattice (identity, q-ary)
            eta = compute_smoothing_parameter(basis_info)
            sigma = sigma_ratio * eta
            logger.info(f"Matrix lattice: eta={eta}, sigma={sigma}")
            results['smoothing_parameter'] = float(eta)
        
        results['sigma'] = float(sigma)
        
        # Run IMHK sampler
        logger.info("Running IMHK sampler...")
        imhk_samples, imhk_metadata = imhk_sampler_wrapper(
            basis_info=basis_info,
            sigma=sigma,
            num_samples=num_samples,
            burn_in=min(1000, num_samples),
            basis_type=basis_type
        )
        
        results['imhk_acceptance_rate'] = imhk_metadata.get('acceptance_rate', 0)
        results['imhk_samples_shape'] = imhk_samples.shape
        
        # Run Klein sampler for comparison (with fewer samples)
        logger.info("Running Klein sampler for comparison...")
        klein_samples, klein_metadata = klein_sampler_wrapper(
            basis_info=basis_info,
            sigma=sigma,
            num_samples=min(100, num_samples),  # Fewer samples for Klein
            basis_type=basis_type
        )
        
        results['klein_samples_shape'] = klein_samples.shape
        
        # Compute TV distance for IMHK samples (if applicable)
        if not isinstance(basis_info, tuple) and imhk_samples.shape[0] > 0:
            logger.info("Computing TV distance...")
            try:
                tv_distance = compute_total_variation_distance(
                    imhk_samples[:min(500, len(imhk_samples))],
                    sigma,
                    basis_info
                )
                results['tv_distance'] = float(tv_distance)
                logger.info(f"TV distance: {tv_distance}")
            except Exception as e:
                logger.warning(f"Could not compute TV distance: {e}")
                results['tv_distance'] = None
        else:
            results['tv_distance'] = None
        
        results['status'] = 'success'
        logger.info(f"Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'failed'
        results['error'] = str(e)
    
    return results

def main():
    """Run comprehensive cryptographic experiments."""
    # Create output directory
    output_dir = Path("crypto_experiment_results")
    output_dir.mkdir(exist_ok=True)
    
    # Define experiment configurations
    experiments = [
        # q-ary lattices with varying dimensions
        ('q-ary', 16, 2.0),
        ('q-ary', 32, 2.0),
        ('q-ary', 64, 2.0),
        
        # NTRU lattices (dimension doesn't affect the polynomial degree)
        ('NTRU', 32, 2.0),
        ('NTRU', 64, 2.0),
        
        # Prime Cyclotomic lattices
        ('PrimeCyclotomic', 32, 2.0),
        ('PrimeCyclotomic', 64, 2.0),
        
        # Baseline identity lattice for comparison
        ('identity', 16, 2.0),
        ('identity', 32, 2.0),
    ]
    
    all_results = []
    
    for basis_type, dim, sigma_ratio in experiments:
        result = run_single_experiment(
            basis_type=basis_type,
            dim=dim,
            sigma_ratio=sigma_ratio,
            num_samples=500  # Moderate number of samples
        )
        all_results.append(result)
        
        # Save intermediate results
        with open(output_dir / f"{basis_type}_{dim}_results.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    # Save all results
    summary = {
        'experiment_date': datetime.now().isoformat(),
        'total_experiments': len(all_results),
        'successful_experiments': sum(1 for r in all_results if r['status'] == 'success'),
        'results': all_results
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    logger.info(f"Total experiments: {summary['total_experiments']}")
    logger.info(f"Successful: {summary['successful_experiments']}")
    
    for result in all_results:
        status = "✓" if result['status'] == 'success' else "✗"
        logger.info(f"{status} {result['basis_type']:15} dim={result['dimension']:3} " +
                   f"acceptance={result.get('imhk_acceptance_rate', 'N/A'):.4f}")
    
    logger.info(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()