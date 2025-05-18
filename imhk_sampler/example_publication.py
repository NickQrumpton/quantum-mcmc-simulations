"""
Example script demonstrating publication-quality IMHK sampler usage.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import our modules
from utils import create_lattice_basis
from samplers import imhk_sampler, klein_sampler
from stats import compute_total_variation_distance
from parameter_config import compute_smoothing_parameter
from diagnostics import compute_ess, compute_autocorrelation
from visualization import plot_samples_2d, plot_trace, plot_autocorrelation


def run_example():
    """Run example demonstrating IMHK sampler functionality."""
    # Create output directory
    output_dir = Path("results/example_publication")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up experiment parameters
    dimension = 2
    basis_type = "identity"
    num_samples = 5000
    burn_in = 1000
    
    # Create lattice basis
    lattice_basis = create_lattice_basis(dimension, basis_type)
    
    # Compute smoothing parameter
    eta = compute_smoothing_parameter(lattice_basis)
    ratio = 1.5  # σ/η ratio
    sigma = ratio * eta
    
    print(f"Running example: dim={dimension}, basis={basis_type}")
    print(f"Smoothing parameter η: {eta:.4f}")
    print(f"Using σ = {sigma:.4f} (ratio={ratio})")
    
    # Run IMHK sampler
    print("\nRunning IMHK sampler...")
    samples_imhk, metadata = imhk_sampler(
        B=lattice_basis,
        sigma=sigma,
        num_samples=num_samples,
        burn_in=burn_in
    )
    
    print(f"IMHK acceptance rate: {metadata['acceptance_rate']:.3f}")
    
    # Run Klein sampler for comparison
    print("\nRunning Klein sampler...")
    samples_klein = klein_sampler(
        B=lattice_basis,
        sigma=sigma,
        num_samples=num_samples
    )
    
    # Compute TV distances
    print("\nComputing TV distances...")
    tv_imhk = compute_total_variation_distance(samples_imhk, sigma, lattice_basis)
    tv_klein = compute_total_variation_distance(samples_klein, sigma, lattice_basis)
    
    print(f"IMHK TV distance: {tv_imhk:.6f}")
    print(f"Klein TV distance: {tv_klein:.6f}")
    print(f"Improvement ratio: {tv_klein/tv_imhk:.2f}x")
    
    # Compute diagnostics
    print("\nComputing diagnostics...")
    ess_imhk = compute_ess(samples_imhk)
    acf_imhk = compute_autocorrelation(samples_imhk)
    
    print(f"IMHK mean ESS: {np.mean(ess_imhk):.1f}")
    
    # Create plots
    print("\nCreating plots...")
    
    # Plot 2D samples
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    plot_samples_2d(samples_imhk[:1000], ax=ax1, title="IMHK Samples")
    plot_samples_2d(samples_klein[:1000], ax=ax2, title="Klein Samples")
    
    plt.tight_layout()
    plt.savefig(output_dir / "samples_comparison.png", dpi=300)
    plt.close()
    
    # Plot trace plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    plot_trace(samples_imhk, ax=ax1, title="IMHK Trace Plot")
    plot_trace(samples_klein, ax=ax2, title="Klein Trace Plot")
    
    plt.tight_layout()
    plt.savefig(output_dir / "trace_comparison.png", dpi=300)
    plt.close()
    
    # Plot autocorrelation
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_autocorrelation(acf_imhk, ax=ax, title="IMHK Autocorrelation")
    plt.savefig(output_dir / "autocorrelation_imhk.png", dpi=300)
    plt.close()
    
    # Save results summary
    results = {
        'dimension': dimension,
        'basis_type': basis_type,
        'sigma': sigma,
        'eta': eta,
        'ratio': ratio,
        'num_samples': num_samples,
        'burn_in': burn_in,
        'imhk_acceptance_rate': metadata['acceptance_rate'],
        'imhk_tv_distance': tv_imhk,
        'klein_tv_distance': tv_klein,
        'improvement_ratio': tv_klein / tv_imhk if tv_imhk > 0 else float('inf'),
        'mean_ess': float(np.mean(ess_imhk))
    }
    
    # Save to JSON
    import json
    with open(output_dir / "results_summary.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_dir}")
    print("Example completed successfully!")


if __name__ == "__main__":
    run_example()