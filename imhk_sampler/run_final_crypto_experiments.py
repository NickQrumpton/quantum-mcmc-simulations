#!/usr/bin/env sage -python
"""
Run final comprehensive experiments with cryptographic lattice bases.
Uses appropriate parameter scaling for research publication.
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from math import sqrt

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import create_lattice_basis
from samplers import imhk_sampler_wrapper, klein_sampler_wrapper
from parameter_config import compute_smoothing_parameter
from stats import compute_total_variation_distance
from diagnostics import compute_ess
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress matplotlib debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def get_appropriate_sigma(basis_info, basis_type, sigma_ratio):
    """Get appropriate sigma value based on basis type and cryptographic standards."""
    if isinstance(basis_info, tuple):
        # Structured lattice (NTRU, PrimeCyclotomic)
        poly_mod, q = basis_info
        # Use standard deviation appropriate for cryptographic applications
        # Based on Falcon/Mitaka parameters
        if basis_type == 'NTRU':
            # Falcon-512/1024 uses sigma around sqrt(q)/50
            sigma = float(sqrt(q) / 50)
        else:  # PrimeCyclotomic
            # Mitaka-style parameters
            sigma = float(sqrt(q) / 40)
    else:
        # Matrix-based lattice (identity, q-ary)
        # Use smoothing parameter approach, but with minimum threshold
        eta = compute_smoothing_parameter(basis_info)
        sigma = max(sigma_ratio * eta, 1.0)  # Ensure sigma >= 1
        
        # For q-ary lattices, ensure sigma is appropriate for the modulus
        if basis_type == 'q-ary':
            # Extract modulus from the lattice (first diagonal element)
            q = abs(basis_info[0, 0])
            sigma = max(sigma, sqrt(q) / 100)
    
    return float(sigma)

def create_visualization(results, output_dir):
    """Create publication-quality visualizations of results."""
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract data for plotting
    basis_types = []
    acceptance_rates = []
    tv_distances = []
    dimensions = []
    
    for result in results:
        if result['status'] == 'success':
            basis_types.append(result['basis_type'])
            acceptance_rates.append(result['imhk_acceptance_rate'])
            tv_distances.append(result.get('tv_distance', 0))
            dimensions.append(result['dimension'])
    
    # Plot 1: Acceptance rates by basis type
    x_pos = np.arange(len(basis_types))
    colors = plt.cm.tab10(np.linspace(0, 1, len(set(basis_types))))
    basis_colors = {bt: colors[i] for i, bt in enumerate(set(basis_types))}
    
    bars1 = ax1.bar(x_pos, acceptance_rates, color=[basis_colors[bt] for bt in basis_types])
    ax1.set_xlabel('Experiment Configuration', fontsize=12)
    ax1.set_ylabel('IMHK Acceptance Rate', fontsize=12)
    ax1.set_title('Acceptance Rates for Cryptographic Lattices', fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{bt}\n(d={d})" for bt, d in zip(basis_types, dimensions)], 
                        rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: TV distances (where available)
    tv_data = [(i, tv) for i, tv in enumerate(tv_distances) if tv > 0]
    if tv_data:
        indices, tv_vals = zip(*tv_data)
        bars2 = ax2.bar(indices, tv_vals, color=[basis_colors[basis_types[i]] for i in indices])
        ax2.set_xlabel('Matrix-based Lattice Configurations', fontsize=12)
        ax2.set_ylabel('Total Variation Distance', fontsize=12)
        ax2.set_title('TV Distance for Matrix-based Lattices', fontsize=14)
        ax2.set_xticks(list(indices))
        ax2.set_xticklabels([f"{basis_types[i]}\n(d={dimensions[i]})" for i in indices], 
                           rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'TV Distance not computed\nfor structured lattices', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'crypto_results_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()


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
        
        # Get appropriate sigma value
        sigma = get_appropriate_sigma(basis_info, basis_type, sigma_ratio)
        results['sigma'] = sigma
        
        if isinstance(basis_info, tuple):
            poly_mod, q = basis_info
            degree = poly_mod.degree()
            logger.info(f"Structured lattice: degree={degree}, q={q}, sigma={sigma}")
            results['polynomial_degree'] = degree
            results['modulus'] = q
        else:
            logger.info(f"Matrix lattice: dimension={dim}, sigma={sigma}")
            # Save smoothing parameter for matrix lattices
            eta = compute_smoothing_parameter(basis_info)
            results['smoothing_parameter'] = float(eta)
        
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
        
        # Compute ESS for IMHK samples
        if imhk_samples.shape[0] > 10:
            try:
                ess = compute_ess(imhk_samples[:, 0])  # ESS for first coordinate
                results['effective_sample_size'] = float(ess)
            except:
                results['effective_sample_size'] = None
        
        # Run Klein sampler for comparison (with fewer samples)
        logger.info("Running Klein sampler for comparison...")
        klein_samples, klein_metadata = klein_sampler_wrapper(
            basis_info=basis_info,
            sigma=sigma,
            num_samples=min(100, num_samples),  # Fewer samples for Klein
            basis_type=basis_type
        )
        
        results['klein_samples_shape'] = klein_samples.shape
        
        # Compute TV distance for matrix-based lattices only
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
    output_dir = Path("crypto_publication_results")
    output_dir.mkdir(exist_ok=True)
    
    # Define experiment configurations
    experiments = [
        # q-ary lattices with varying dimensions
        ('q-ary', 16, 2.0),
        ('q-ary', 32, 2.0),
        ('q-ary', 64, 3.0),  # Higher ratio for larger dimension
        
        # NTRU lattices (dimension determines polynomial degree)
        ('NTRU', 32, 2.0),  # Will use N=512
        ('NTRU', 64, 2.0),  # Will use N=1024
        
        # Prime Cyclotomic lattices
        ('PrimeCyclotomic', 32, 2.0),
        ('PrimeCyclotomic', 64, 2.0),
        
        # Baseline identity lattice for comparison
        ('identity', 16, 2.0),
        ('identity', 32, 2.0),
        ('identity', 64, 2.0),
    ]
    
    all_results = []
    
    for basis_type, dim, sigma_ratio in experiments:
        result = run_single_experiment(
            basis_type=basis_type,
            dim=dim,
            sigma_ratio=sigma_ratio,
            num_samples=1000  # Sufficient samples for publication
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
    
    # Create visualizations
    create_visualization(all_results, output_dir)
    
    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    logger.info(f"Total experiments: {summary['total_experiments']}")
    logger.info(f"Successful: {summary['successful_experiments']}")
    logger.info("\nDETAILED RESULTS:")
    logger.info("-"*80)
    logger.info(f"{'Basis Type':15} {'Dim':>5} {'Sigma':>12} {'Acc. Rate':>10} {'ESS':>8} {'TV Dist':>10}")
    logger.info("-"*80)
    
    for result in all_results:
        if result['status'] == 'success':
            basis = result['basis_type']
            dim = result['dimension']
            sigma = result['sigma']
            acc_rate = result.get('imhk_acceptance_rate', 'N/A')
            ess = result.get('effective_sample_size', 'N/A')
            tv_dist = result.get('tv_distance', 'N/A')
            
            logger.info(f"{basis:15} {dim:5d} {sigma:12.6f} {acc_rate:10.4f} "
                       f"{ess:>8} {tv_dist if tv_dist != 'N/A' else 'N/A':>10}")
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"Visualization saved as crypto_results_visualization.png")

if __name__ == "__main__":
    main()