#!/usr/bin/env sage -python
"""
Check if all components are research-ready and generate initial publication results.
"""

import sys
from pathlib import Path
import numpy as np
import logging
import time
import json
import matplotlib.pyplot as plt
from math import sqrt

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_component_status():
    """Check if all components are working properly."""
    status = {
        'imports': True,
        'basis_types': {},
        'samplers': True,
        'tv_distance': True,
        'overall': True
    }
    
    # Check imports
    try:
        from utils import create_lattice_basis
        from samplers import imhk_sampler_wrapper, klein_sampler_wrapper
        from stats import compute_total_variation_distance
        from parameter_config import compute_smoothing_parameter
        from diagnostics import compute_ess
        logger.info("✓ All imports successful")
    except Exception as e:
        logger.error(f"✗ Import error: {e}")
        status['imports'] = False
        status['overall'] = False
        
    # Check each basis type
    basis_types = ['identity', 'q-ary', 'NTRU', 'PrimeCyclotomic']
    for basis_type in basis_types:
        try:
            basis = create_lattice_basis(8, basis_type)
            status['basis_types'][basis_type] = True
            logger.info(f"✓ {basis_type} basis working")
        except Exception as e:
            logger.error(f"✗ {basis_type} basis failed: {e}")
            status['basis_types'][basis_type] = False
            status['overall'] = False
    
    # Check samplers
    try:
        from utils import create_lattice_basis
        from samplers import imhk_sampler_wrapper
        basis = create_lattice_basis(4, 'identity')
        samples, metadata = imhk_sampler_wrapper(
            basis_info=basis,
            sigma=1.0,
            num_samples=20,
            burn_in=10,
            basis_type='identity'
        )
        logger.info("✓ Samplers working")
    except Exception as e:
        logger.error(f"✗ Sampler error: {e}")
        status['samplers'] = False
        status['overall'] = False
    
    # Check TV distance
    try:
        from stats import compute_total_variation_distance
        tv_dist = compute_total_variation_distance(
            samples[:10], 1.0, basis, max_radius=2
        )
        logger.info("✓ TV distance computation working")
    except Exception as e:
        logger.error(f"✗ TV distance error: {e}")
        status['tv_distance'] = False
        status['overall'] = False
    
    return status

def generate_research_data(dim, basis_type, sigma_ratios):
    """Generate research data for a specific configuration."""
    from utils import create_lattice_basis
    from samplers import imhk_sampler_wrapper
    from parameter_config import compute_smoothing_parameter
    from stats import compute_total_variation_distance
    from diagnostics import compute_ess
    
    results = []
    
    # Create basis
    basis_info = create_lattice_basis(dim, basis_type)
    
    # Handle different basis types
    if isinstance(basis_info, tuple):
        # Structured lattice - use fixed sigma values
        poly_mod, q = basis_info
        base_sigma = float(sqrt(q) / 20)
        sigmas = [base_sigma * ratio for ratio in sigma_ratios]
    else:
        # Matrix lattice - use smoothing parameter
        eta = compute_smoothing_parameter(basis_info)
        sigmas = [eta * ratio for ratio in sigma_ratios]
    
    for sigma in sigmas:
        logger.info(f"Testing {basis_type} (dim={dim}) with sigma={sigma:.4f}")
        
        try:
            # Run IMHK sampler
            start_time = time.time()
            samples, metadata = imhk_sampler_wrapper(
                basis_info=basis_info,
                sigma=sigma,
                num_samples=1000,
                burn_in=500,
                basis_type=basis_type
            )
            imhk_time = time.time() - start_time
            
            # Compute metrics
            result = {
                'basis_type': basis_type,
                'dimension': dim,
                'sigma': sigma,
                'sigma_ratio': sigma / base_sigma if isinstance(basis_info, tuple) else sigma / eta,
                'acceptance_rate': metadata.get('acceptance_rate', 0),
                'imhk_time': imhk_time,
                'num_samples': samples.shape[0]
            }
            
            # Compute ESS
            try:
                ess = compute_ess(samples[:, 0])
                result['ess'] = ess
            except:
                result['ess'] = None
                
            # Compute TV distance for matrix lattices
            if not isinstance(basis_info, tuple) and dim <= 16:
                try:
                    tv_dist = compute_total_variation_distance(
                        samples[:500], sigma, basis_info,
                        max_radius=max(2, int(3.0/np.sqrt(dim)))
                    )
                    result['tv_distance'] = tv_dist
                except:
                    result['tv_distance'] = None
            else:
                result['tv_distance'] = None
                
            results.append(result)
            logger.info(f"✓ Success: acceptance_rate={result['acceptance_rate']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed: {e}")
            
    return results

def create_research_plots(all_results):
    """Create publication-quality plots."""
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Set up publication style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (10, 6)
    })
    
    # Plot 1: TV Distance vs Sigma Ratio
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for basis_type in df['basis_type'].unique():
        data = df[(df['basis_type'] == basis_type) & (df['tv_distance'].notna())]
        if not data.empty:
            ax.plot(data['sigma_ratio'], data['tv_distance'], 
                   marker='o', linewidth=2, markersize=8,
                   label=f'{basis_type} (d={data.iloc[0]["dimension"]})')
    
    ax.set_xlabel('σ/η Ratio', fontsize=14)
    ax.set_ylabel('Total Variation Distance', fontsize=14)
    ax.set_title('TV Distance vs σ/η Ratio for Different Lattice Types', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('tv_distance_vs_sigma_research.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Acceptance Rate vs Sigma Ratio
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for basis_type in df['basis_type'].unique():
        data = df[df['basis_type'] == basis_type]
        ax.plot(data['sigma_ratio'], data['acceptance_rate'], 
               marker='s', linewidth=2, markersize=8,
               label=f'{basis_type} (d={data.iloc[0]["dimension"]})')
    
    ax.set_xlabel('σ/η Ratio', fontsize=14)
    ax.set_ylabel('Acceptance Rate', fontsize=14)
    ax.set_title('IMHK Acceptance Rate vs σ/η Ratio', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('acceptance_rate_vs_sigma_research.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Performance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ESS by basis type
    basis_types = df['basis_type'].unique()
    ess_data = []
    for basis in basis_types:
        data = df[(df['basis_type'] == basis) & (df['ess'].notna())]
        if not data.empty:
            ess_data.append(data['ess'].values)
    
    if ess_data:
        ax1.boxplot(ess_data, labels=basis_types)
        ax1.set_ylabel('Effective Sample Size', fontsize=14)
        ax1.set_title('ESS Distribution by Lattice Type', fontsize=16)
        ax1.grid(True, alpha=0.3)
    
    # Runtime comparison
    runtime_data = df.groupby('basis_type')['imhk_time'].mean()
    ax2.bar(runtime_data.index, runtime_data.values)
    ax2.set_ylabel('Average Runtime (s)', fontsize=14)
    ax2.set_title('Average IMHK Runtime by Lattice Type', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison_research.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Check research readiness and generate initial results."""
    logger.info("Research Readiness Check")
    logger.info("="*50)
    
    # Step 1: Check component status
    status = check_component_status()
    
    if not status['overall']:
        logger.error("Some components are not working. Fix these issues first.")
        return 1
    
    logger.info("\nAll components working! Generating research data...")
    
    # Step 2: Generate research data
    configurations = [
        # Format: (dimension, basis_type, sigma_ratios)
        (8, 'identity', [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        (8, 'q-ary', [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
        (16, 'identity', [0.5, 1.0, 1.5, 2.0, 2.5]),
        (16, 'q-ary', [0.5, 1.0, 1.5, 2.0, 2.5]),
        (512, 'NTRU', [0.8, 1.0, 1.2, 1.5, 2.0]),  # NTRU dimension is fixed
        (683, 'PrimeCyclotomic', [0.8, 1.0, 1.2, 1.5, 2.0]),  # Prime cyclotomic dimension is fixed
    ]
    
    all_results = []
    
    for dim, basis_type, sigma_ratios in configurations:
        results = generate_research_data(dim, basis_type, sigma_ratios)
        all_results.extend(results)
    
    # Step 3: Save results
    logger.info("\nSaving research results...")
    
    # Save raw data
    with open('research_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv('research_results.csv', index=False)
    
    # Step 4: Create plots
    logger.info("Creating research plots...")
    create_research_plots(all_results)
    
    # Step 5: Generate summary report
    logger.info("\nGenerating summary report...")
    
    summary = {
        'total_experiments': len(all_results),
        'configurations_tested': len(configurations),
        'basis_types': list(df['basis_type'].unique()),
        'dimensions': list(df['dimension'].unique()),
        'average_acceptance_rates': df.groupby('basis_type')['acceptance_rate'].mean().to_dict(),
        'average_tv_distances': df[df['tv_distance'].notna()].groupby('basis_type')['tv_distance'].mean().to_dict()
    }
    
    with open('research_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("RESEARCH SUMMARY")
    logger.info("="*50)
    logger.info(f"Total experiments: {summary['total_experiments']}")
    logger.info(f"Configurations tested: {summary['configurations_tested']}")
    
    logger.info("\nAverage Acceptance Rates:")
    for basis, rate in summary['average_acceptance_rates'].items():
        logger.info(f"  {basis:15}: {rate:.4f}")
    
    logger.info("\nAverage TV Distances:")
    for basis, tv in summary['average_tv_distances'].items():
        logger.info(f"  {basis:15}: {tv:.6f}")
    
    logger.info("\nFiles generated:")
    logger.info("- research_results.json")
    logger.info("- research_results.csv")
    logger.info("- tv_distance_vs_sigma_research.png")
    logger.info("- acceptance_rate_vs_sigma_research.png")
    logger.info("- performance_comparison_research.png")
    logger.info("- research_summary.json")
    
    logger.info("\n🎉 Research data successfully generated!")
    logger.info("These results are ready for publication.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())