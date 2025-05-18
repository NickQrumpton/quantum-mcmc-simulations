#!/usr/bin/env sage -python
"""
Generate comprehensive publication-quality results for cryptographic lattice bases.
This script implements the researcher's specifications for lattice cryptography experiments.
"""

import sys
from pathlib import Path
import numpy as np
import json
import pandas as pd
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Publication-quality plot settings
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (10, 8),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def run_experiment(basis_type, dim, sigma_ratio=2.0, num_samples=2000):
    """Run a single cryptographic lattice experiment."""
    logger.info(f"Running {basis_type} experiment (d={dim})")
    
    try:
        # Create lattice basis
        basis_info = create_lattice_basis(dim, basis_type)
        
        # Determine appropriate sigma
        if isinstance(basis_info, tuple):
            # Structured lattice (NTRU, PrimeCyclotomic)
            poly_mod, q = basis_info
            if basis_type == 'NTRU':
                sigma = float(sqrt(q) / 50)  # Falcon-style parameters
            else:
                sigma = float(sqrt(q) / 40)  # Mitaka-style parameters
            degree = poly_mod.degree()
        else:
            # Matrix lattice (identity, q-ary)
            eta = compute_smoothing_parameter(basis_info)
            sigma = max(sigma_ratio * eta, 1.0)
            if basis_type == 'q-ary':
                q = abs(basis_info[0, 0])
                sigma = max(sigma, sqrt(q) / 100)
            degree = None
        
        # Run IMHK sampler
        imhk_samples, imhk_metadata = imhk_sampler_wrapper(
            basis_info=basis_info,
            sigma=sigma,
            num_samples=num_samples,
            burn_in=min(1000, num_samples//2),
            basis_type=basis_type
        )
        
        # Compute metrics
        acceptance_rate = imhk_metadata.get('acceptance_rate', 0)
        
        # Compute ESS for first coordinate
        ess = None
        if imhk_samples.shape[0] > 100:
            try:
                ess = compute_ess(imhk_samples[:, 0])
            except:
                pass
        
        # Compute TV distance for matrix lattices
        tv_distance = None
        if not isinstance(basis_info, tuple) and imhk_samples.shape[0] > 100:
            try:
                tv_distance = compute_total_variation_distance(
                    imhk_samples[:min(1000, len(imhk_samples))],
                    sigma,
                    basis_info
                )
            except:
                pass
        
        return {
            'basis_type': basis_type,
            'dimension': dim,
            'sigma': float(sigma),
            'acceptance_rate': float(acceptance_rate),
            'ess': float(ess) if ess else None,
            'tv_distance': float(tv_distance) if tv_distance else None,
            'polynomial_degree': degree,
            'num_samples': num_samples
        }
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        return None

def generate_comparison_plots(results_df, output_dir):
    """Generate publication-quality comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Acceptance rates by basis type
    ax1 = axes[0, 0]
    basis_grouped = results_df.groupby('basis_type')['acceptance_rate'].mean()
    colors = plt.cm.tab10(np.linspace(0, 1, len(basis_grouped)))
    bars = ax1.bar(basis_grouped.index, basis_grouped.values, color=colors)
    ax1.set_xlabel('Basis Type')
    ax1.set_ylabel('Mean Acceptance Rate')
    ax1.set_title('IMHK Acceptance Rates by Lattice Type')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Acceptance rate vs dimension
    ax2 = axes[0, 1]
    for i, basis_type in enumerate(results_df['basis_type'].unique()):
        data = results_df[results_df['basis_type'] == basis_type]
        ax2.plot(data['dimension'], data['acceptance_rate'], 
                marker='o', linewidth=2, markersize=8,
                label=basis_type, color=colors[i])
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Acceptance Rate')
    ax2.set_title('Acceptance Rate vs Dimension')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: TV distance for matrix lattices
    ax3 = axes[1, 0]
    tv_data = results_df[results_df['tv_distance'].notna()]
    if not tv_data.empty:
        for i, basis_type in enumerate(tv_data['basis_type'].unique()):
            data = tv_data[tv_data['basis_type'] == basis_type]
            ax3.scatter(data['dimension'], data['tv_distance'],
                       s=100, marker='o', label=basis_type,
                       color=colors[list(results_df['basis_type'].unique()).index(basis_type)])
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Total Variation Distance')
        ax3.set_title('TV Distance for Matrix-based Lattices')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No TV distance data\n(structured lattices only)', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Effective sample size
    ax4 = axes[1, 1]
    ess_data = results_df[results_df['ess'].notna()]
    if not ess_data.empty:
        for i, basis_type in enumerate(ess_data['basis_type'].unique()):
            data = ess_data[ess_data['basis_type'] == basis_type]
            ax4.scatter(data['dimension'], data['ess'],
                       s=100, marker='s', label=basis_type,
                       color=colors[list(results_df['basis_type'].unique()).index(basis_type)])
        ax4.set_xlabel('Dimension')
        ax4.set_ylabel('Effective Sample Size')
        ax4.set_title('ESS by Lattice Type and Dimension')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'crypto_lattice_comparison.png', dpi=300)
    plt.close()

def main():
    """Generate comprehensive cryptographic lattice results."""
    output_dir = Path("crypto_publication_results")
    output_dir.mkdir(exist_ok=True)
    
    # Comprehensive experiment configurations
    experiments = [
        # Identity baseline
        ('identity', 16, 2.0),
        ('identity', 32, 2.0),
        ('identity', 64, 2.0),
        
        # q-ary lattices (LWE-based)
        ('q-ary', 16, 2.0),
        ('q-ary', 32, 2.0),
        ('q-ary', 64, 2.5),
        ('q-ary', 128, 3.0),
        
        # NTRU lattices (polynomial-based)
        ('NTRU', 512, 2.0),   # Falcon-512 dimension
        ('NTRU', 1024, 2.0),  # Falcon-1024 dimension
        
        # Prime Cyclotomic (Mitaka-style)
        ('PrimeCyclotomic', 683, 2.0),
    ]
    
    results = []
    
    # Run experiments
    for basis_type, dim, sigma_ratio in experiments:
        result = run_experiment(basis_type, dim, sigma_ratio, num_samples=1000)
        if result:
            results.append(result)
            # Save individual result
            with open(output_dir / f"{basis_type}_{dim}_result.json", 'w') as f:
                json.dump(result, f, indent=2)
    
    # Create results dataframe
    df = pd.DataFrame(results)
    
    # Save comprehensive results
    df.to_csv(output_dir / 'all_results.csv', index=False)
    
    # Generate plots
    generate_comparison_plots(df, output_dir)
    
    # Create summary statistics
    summary = {
        'experiment_date': datetime.now().isoformat(),
        'total_experiments': len(results),
        'configurations': experiments,
        'mean_acceptance_by_type': df.groupby('basis_type')['acceptance_rate'].mean().to_dict(),
        'algorithms': {
            'identity': 'Standard discrete Gaussian baseline',
            'q-ary': 'LWE-based lattices with prime modulus',
            'NTRU': 'Falcon NTRU parameters (NIST standard)',
            'PrimeCyclotomic': 'Mitaka-style prime cyclotomic lattice'
        }
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create LaTeX table for publication
    latex_table = df.to_latex(
        index=False,
        columns=['basis_type', 'dimension', 'sigma', 'acceptance_rate', 'tv_distance'],
        float_format='%.4f',
        caption='IMHK Sampler Performance on Cryptographic Lattices',
        label='tab:crypto_results'
    )
    
    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write(latex_table)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("CRYPTOGRAPHIC LATTICE EXPERIMENT SUMMARY")
    logger.info("="*60)
    logger.info(f"Total experiments: {len(results)}")
    logger.info("\nAcceptance Rates by Type:")
    for basis_type in df['basis_type'].unique():
        data = df[df['basis_type'] == basis_type]
        logger.info(f"  {basis_type:15} {data['acceptance_rate'].mean():.4f} ± {data['acceptance_rate'].std():.4f}")
    
    logger.info(f"\nResults saved to: {output_dir}/")
    logger.info("Files generated:")
    logger.info("  - all_results.csv: Complete results table")
    logger.info("  - crypto_lattice_comparison.png: Visualization plots")
    logger.info("  - summary.json: Experiment summary")
    logger.info("  - results_table.tex: LaTeX table for publication")

if __name__ == "__main__":
    main()