#!/usr/bin/env python3
"""
Generate publication-quality results for comparing Total Variation distance
between IMHK and Klein algorithms across different sigma values.

This script runs comprehensive experiments suitable for academic publication.
"""

from experiments import compare_tv_distance_vs_sigma
import logging
import time
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('publication_tv_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("publication_results")

def run_comprehensive_comparison():
    """Run comprehensive TV distance comparison for publication."""
    
    start_time = time.time()
    
    # Publication-quality parameters
    dimensions = [8, 16, 32, 64]  # Standard cryptographic dimensions
    basis_types = ['identity', 'skewed', 'ill-conditioned']  # All implemented basis types
    sigma_eta_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]  # Comprehensive range
    num_samples = 5000  # Sufficient for statistical accuracy
    
    output_dir = Path("results/publication/tv_distance_comparison")
    
    logger.info("Starting comprehensive TV distance comparison")
    logger.info(f"Parameters:")
    logger.info(f"  Dimensions: {dimensions}")
    logger.info(f"  Basis types: {basis_types}")
    logger.info(f"  Sigma/eta ratios: {sigma_eta_ratios}")
    logger.info(f"  Number of samples: {num_samples}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Run the comprehensive comparison
    results = compare_tv_distance_vs_sigma(
        dimensions=dimensions,
        basis_types=basis_types,
        sigma_eta_ratios=sigma_eta_ratios,
        num_samples=num_samples,
        plot_results=True,
        output_dir=output_dir
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Completed in {elapsed_time/3600:.2f} hours")
    
    # Generate summary statistics
    generate_summary_statistics(results, output_dir)
    
    return results

def generate_summary_statistics(results, output_dir):
    """Generate summary statistics from the results."""
    
    summary = {
        'total_experiments': len(results),
        'successful_experiments': 0,
        'failed_experiments': 0,
        'dimensions_analyzed': [],
        'basis_types_analyzed': [],
        'best_improvements': [],
        'configuration_summaries': {}
    }
    
    # Analyze results
    for key, result in results.items():
        dim, basis_type, ratio = key
        
        if dim not in summary['dimensions_analyzed']:
            summary['dimensions_analyzed'].append(dim)
        if basis_type not in summary['basis_types_analyzed']:
            summary['basis_types_analyzed'].append(basis_type)
        
        if 'error' in result:
            summary['failed_experiments'] += 1
        else:
            summary['successful_experiments'] += 1
            
            # Track best improvements
            if 'tv_ratio' in result and result['tv_ratio'] is not None:
                improvement = {
                    'dimension': dim,
                    'basis_type': basis_type,
                    'sigma_eta_ratio': ratio,
                    'tv_ratio': result['tv_ratio'],
                    'imhk_tv': result['imhk_tv_distance'],
                    'klein_tv': result['klein_tv_distance']
                }
                summary['best_improvements'].append(improvement)
        
        # Configuration-specific summaries
        config_name = f"dim{dim}_{basis_type}"
        if config_name not in summary['configuration_summaries']:
            summary['configuration_summaries'][config_name] = {
                'dimension': dim,
                'basis_type': basis_type,
                'tv_ratios': [],
                'best_ratio': None,
                'worst_ratio': None,
                'average_ratio': None
            }
        
        if 'tv_ratio' in result and result['tv_ratio'] is not None:
            summary['configuration_summaries'][config_name]['tv_ratios'].append(result['tv_ratio'])
    
    # Sort best improvements
    summary['best_improvements'].sort(key=lambda x: x['tv_ratio'])
    summary['best_improvements'] = summary['best_improvements'][:10]  # Top 10
    
    # Calculate configuration summaries
    for config_name, config_data in summary['configuration_summaries'].items():
        if config_data['tv_ratios']:
            config_data['best_ratio'] = min(config_data['tv_ratios'])
            config_data['worst_ratio'] = max(config_data['tv_ratios'])
            config_data['average_ratio'] = sum(config_data['tv_ratios']) / len(config_data['tv_ratios'])
            config_data['tv_ratios'] = []  # Clear to save space in JSON
    
    # Save summary
    summary_path = output_dir / "summary_statistics.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary statistics to {summary_path}")
    
    # Generate key findings
    generate_key_findings(summary, output_dir)

def generate_key_findings(summary, output_dir):
    """Generate a key findings report from the summary."""
    
    findings_path = output_dir / "key_findings.txt"
    
    with open(findings_path, 'w') as f:
        f.write("KEY FINDINGS: TV Distance Comparison (IMHK vs Klein)\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total experiments: {summary['total_experiments']}\n")
        f.write(f"Successful: {summary['successful_experiments']}\n")
        f.write(f"Failed: {summary['failed_experiments']}\n\n")
        
        f.write("DIMENSIONS ANALYZED:\n")
        for dim in sorted(summary['dimensions_analyzed']):
            f.write(f"  - {dim}D\n")
        f.write("\n")
        
        f.write("BASIS TYPES ANALYZED:\n")
        for basis in sorted(summary['basis_types_analyzed']):
            f.write(f"  - {basis}\n")
        f.write("\n")
        
        f.write("TOP 10 IMPROVEMENTS (IMHK over Klein):\n")
        for i, improvement in enumerate(summary['best_improvements'][:10], 1):
            f.write(f"{i}. Dim={improvement['dimension']}, "
                   f"Basis={improvement['basis_type']}, "
                   f"σ/η={improvement['sigma_eta_ratio']}, "
                   f"TV Ratio={improvement['tv_ratio']:.6f}\n")
        f.write("\n")
        
        f.write("CONFIGURATION SUMMARIES:\n")
        for config_name, config_data in sorted(summary['configuration_summaries'].items()):
            if config_data['best_ratio'] is not None:
                f.write(f"\n{config_name}:\n")
                f.write(f"  Best TV ratio: {config_data['best_ratio']:.6f}\n")
                f.write(f"  Worst TV ratio: {config_data['worst_ratio']:.6f}\n")
                f.write(f"  Average TV ratio: {config_data['average_ratio']:.6f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("CONCLUSIONS:\n")
        
        # Generate automatic conclusions
        best_overall = summary['best_improvements'][0] if summary['best_improvements'] else None
        if best_overall:
            f.write(f"- Best improvement: {(1-best_overall['tv_ratio'])*100:.1f}% "
                   f"(Dim={best_overall['dimension']}, "
                   f"Basis={best_overall['basis_type']}, "
                   f"σ/η={best_overall['sigma_eta_ratio']})\n")
        
        # Average improvements by basis type
        basis_improvements = {}
        for config_name, config_data in summary['configuration_summaries'].items():
            basis_type = config_data['basis_type']
            if config_data['average_ratio'] is not None:
                if basis_type not in basis_improvements:
                    basis_improvements[basis_type] = []
                basis_improvements[basis_type].append(config_data['average_ratio'])
        
        f.write("\nAverage improvement by basis type:\n")
        for basis_type, ratios in basis_improvements.items():
            avg_ratio = sum(ratios) / len(ratios)
            f.write(f"  - {basis_type}: {(1-avg_ratio)*100:.1f}% improvement\n")
    
    logger.info(f"Generated key findings report at {findings_path}")

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Starting publication-quality TV distance comparison")
    logger.info("=" * 50)
    
    try:
        results = run_comprehensive_comparison()
        logger.info("Successfully completed all experiments")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
    
    logger.info("=" * 50)
    logger.info("All tasks completed")
    logger.info("=" * 50)