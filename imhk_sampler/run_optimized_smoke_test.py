#!/usr/bin/env sage -python
"""
Optimized smoke test for IMHK sampler with performance improvements.

This test includes:
- Lower dimensions for faster runtime
- Progress logging
- Safe interrupt handling
- Diagnostic information
"""

import sys
import argparse
import logging
import time
import signal
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import create_lattice_basis
from samplers import imhk_sampler_wrapper
from stats import compute_total_variation_distance
from parameter_config import compute_smoothing_parameter
import stats_optimized

# Configure logging with more detail
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Interrupt handling
_interrupted = False

def signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    logger.warning("Interrupt received, finishing current experiment...")

signal.signal(signal.SIGINT, signal_handler)


def run_single_experiment(config):
    """Run a single experiment with progress tracking."""
    global _interrupted
    
    if _interrupted:
        return None
    
    experiment_id = f"{config['basis_type']}_d{config['dimension']}_r{config['ratio']}"
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting experiment: {experiment_id}")
    logger.info(f"Configuration: {config}")
    
    start_time = time.time()
    results = {
        'config': config,
        'start_time': start_time,
        'status': 'running'
    }
    
    try:
        # Create lattice basis
        basis_info = create_lattice_basis(config['dimension'], config['basis_type'])
        
        # Determine sigma
        if isinstance(basis_info, tuple):
            # Structured lattice
            poly_mod, q = basis_info
            sigma = float(np.sqrt(q) / 20)
            logger.info(f"Structured lattice: q={q}, sigma={sigma}")
        else:
            # Matrix lattice
            eta = compute_smoothing_parameter(basis_info)
            sigma = config['ratio'] * eta
            logger.info(f"Matrix lattice: eta={eta}, sigma={sigma}")
        
        results['sigma'] = sigma
        
        # Estimate TV distance computation complexity
        if not isinstance(basis_info, tuple):
            sample_size_rec = stats_optimized.estimate_tv_distance_sample_size(
                config['dimension'], sigma
            )
            logger.info(f"Recommended sample size: {sample_size_rec}")
        
        # Run sampler
        logger.info(f"Running {config['basis_type']} sampler...")
        samples, metadata = imhk_sampler_wrapper(
            basis_info=basis_info,
            sigma=sigma,
            num_samples=config['num_samples'],
            burn_in=config['burn_in'],
            basis_type=config['basis_type']
        )
        
        results['acceptance_rate'] = metadata.get('acceptance_rate', 0)
        results['samples_shape'] = samples.shape
        
        # Compute TV distance for matrix lattices only
        if not isinstance(basis_info, tuple) and config['compute_tv']:
            logger.info("Computing TV distance with optimizations...")
            
            # Use limited samples for TV distance to speed up computation
            tv_samples = samples[:min(500, len(samples))]
            
            tv_distance = compute_total_variation_distance(
                tv_samples, 
                sigma, 
                basis_info,
                max_radius=max(2, int(3.0 / np.sqrt(config['dimension']))),
                convergence_threshold=1e-3,
                progress_interval=2.0,
                max_points=5000,
                adaptive_sampling=True
            )
            
            results['tv_distance'] = tv_distance
            
            # Run diagnostics
            diagnostics = stats_optimized.diagnose_sampling_quality(
                tv_samples, basis_info, sigma
            )
            results['diagnostics'] = diagnostics
        else:
            results['tv_distance'] = None
            logger.info("Skipping TV distance (structured lattice or disabled)")
        
        elapsed_time = time.time() - start_time
        results['elapsed_time'] = elapsed_time
        results['status'] = 'success'
        
        logger.info(f"Experiment completed in {elapsed_time:.2f}s")
        logger.info(f"Results: acceptance_rate={results['acceptance_rate']:.4f}, "
                   f"tv_distance={results.get('tv_distance', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'failed'
        results['error'] = str(e)
    
    return results


def main():
    """Run optimized smoke test."""
    parser = argparse.ArgumentParser(description='Optimized IMHK smoke test')
    parser.add_argument('--dimensions', type=int, nargs='+', default=[8, 16])
    parser.add_argument('--basis-types', nargs='+', 
                       default=['identity', 'q-ary', 'NTRU'])
    parser.add_argument('--ratios', type=float, nargs='+', default=[1.0, 1.5, 2.0])
    parser.add_argument('--num-samples', type=int, nargs='+', default=[100, 500, 1000])
    parser.add_argument('--burn-in', type=int, default=100)
    parser.add_argument('--compute-tv', action='store_true', 
                       help='Compute TV distance (slower)')
    
    args = parser.parse_args()
    
    logger.info("Starting optimized smoke test")
    logger.info(f"Configuration: {vars(args)}")
    
    # Generate experiment configurations
    experiments = []
    for basis_type in args.basis_types:
        for dim in args.dimensions:
            for ratio in args.ratios:
                for num_samples in args.num_samples:
                    config = {
                        'basis_type': basis_type,
                        'dimension': dim,
                        'ratio': ratio,
                        'num_samples': num_samples,
                        'burn_in': args.burn_in,
                        'compute_tv': args.compute_tv
                    }
                    experiments.append(config)
    
    logger.info(f"Total experiments: {len(experiments)}")
    
    # Run experiments
    results = []
    total_start = time.time()
    
    for i, config in enumerate(experiments):
        if _interrupted:
            logger.warning("Test interrupted by user")
            break
            
        logger.info(f"\nExperiment {i+1}/{len(experiments)}")
        result = run_single_experiment(config)
        if result:
            results.append(result)
    
    total_elapsed = time.time() - total_start
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Total time: {total_elapsed:.2f}s")
    logger.info(f"Experiments run: {len(results)}/{len(experiments)}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if successful:
        logger.info("\nPerformance Summary:")
        for result in successful:
            config = result['config']
            logger.info(f"  {config['basis_type']}_d{config['dimension']}_n{config['num_samples']}: "
                       f"{result['elapsed_time']:.2f}s, "
                       f"acc_rate={result['acceptance_rate']:.4f}")
    
    # Save results
    output_file = Path("optimized_smoke_test_results.json")
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_file}")
    
    # Final recommendations
    logger.info("\nRecommendations:")
    if any(r['config']['dimension'] > 16 for r in results):
        logger.info("- High dimensions detected. Consider using adaptive sampling.")
    if total_elapsed > 300:
        logger.info("- Long runtime detected. Consider reducing dimensions or samples.")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())