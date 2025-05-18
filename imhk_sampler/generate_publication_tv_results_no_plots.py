#!/usr/bin/env python3
"""
Generate publication-quality results for comparing Total Variation distance
between IMHK and Klein algorithms across different sigma values.
This version skips individual experiment plots to avoid visualization errors.
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
        logging.FileHandler('publication_tv_results_no_plots.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("publication_results")

def run_simpler_experiments():
    """Run a simpler set of experiments without plots."""
    
    start_time = time.time()
    
    # Smaller scale for testing
    dimensions = [4, 8, 16]  # Smaller dimensions
    basis_types = ['identity', 'skewed', 'ill-conditioned']
    sigma_eta_ratios = [0.5, 1.0, 2.0, 4.0]  # Fewer ratios
    num_samples = 500  # Fewer samples
    
    output_dir = Path("results/publication/tv_distance_comparison_simplified")
    
    logger.info("Starting simplified TV distance comparison (no plots)")
    logger.info(f"Parameters:")
    logger.info(f"  Dimensions: {dimensions}")
    logger.info(f"  Basis types: {basis_types}")
    logger.info(f"  Sigma/eta ratios: {sigma_eta_ratios}")
    logger.info(f"  Number of samples: {num_samples}")
    logger.info(f"  Output directory: {output_dir}")
    
    # Run the comparison with plot_results=False
    results = compare_tv_distance_vs_sigma(
        dimensions=dimensions,
        basis_types=basis_types,
        sigma_eta_ratios=sigma_eta_ratios,
        num_samples=num_samples,
        plot_results=False,  # Disable plots to avoid errors
        output_dir=output_dir
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Completed in {elapsed_time/60:.2f} minutes")
    
    # Generate summary report
    generate_summary_report(results, output_dir)
    
    return results

def generate_summary_report(results, output_dir):
    """Generate a summary report from results."""
    
    report_path = output_dir / "summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("Summary Report: TV Distance Comparison\n")
        f.write("=" * 40 + "\n\n")
        
        successful = 0
        failed = 0
        
        for key, result in results.items():
            if 'error' in result:
                failed += 1
            else:
                successful += 1
        
        f.write(f"Total experiments: {len(results)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")
        
        # Results by configuration
        f.write("Results by Configuration:\n")
        f.write("-" * 40 + "\n\n")
        
        for key, result in sorted(results.items()):
            dim, basis_type, ratio = key
            f.write(f"Dimension: {dim}, Basis: {basis_type}, σ/η ratio: {ratio}\n")
            
            if 'error' in result:
                f.write(f"  ERROR: {result['error']}\n")
            else:
                imhk_tv = result.get('imhk_tv_distance', 'N/A')
                klein_tv = result.get('klein_tv_distance', 'N/A')
                tv_ratio = result.get('tv_ratio', 'N/A')
                acceptance = result.get('imhk_acceptance_rate', 'N/A')
                
                if imhk_tv != 'N/A':
                    f.write(f"  IMHK TV distance: {imhk_tv:.6f}\n")
                if klein_tv != 'N/A':
                    f.write(f"  Klein TV distance: {klein_tv:.6f}\n")
                if tv_ratio != 'N/A':
                    f.write(f"  TV ratio (IMHK/Klein): {tv_ratio:.6f}\n")
                if acceptance != 'N/A':
                    f.write(f"  IMHK acceptance rate: {acceptance:.4f}\n")
            
            f.write("\n")
    
    logger.info(f"Generated summary report at {report_path}")

if __name__ == "__main__":
    logger.info("Starting simplified experiments (no individual plots)")
    
    try:
        results = run_simpler_experiments()
        logger.info("Successfully completed experiments")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
    
    logger.info("All tasks completed")