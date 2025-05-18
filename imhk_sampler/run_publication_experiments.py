#!/usr/bin/env python3
"""
Run publication-quality experiments for IMHK sampler.

This script runs the comprehensive experiments as specified in the refactoring requirements.
"""

import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/experiment_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.report import ExperimentRunner


def main():
    """Main entry point for running publication experiments."""
    logger.info("Starting publication-quality IMHK experiments")
    
    # Create experiment runner with custom parameters if needed
    runner = ExperimentRunner(
        output_dir="results/publication_experiments"
    )
    
    # Customize experiment parameters
    runner.num_chains = 5  # Number of independent chains
    runner.num_samples = 10000  # Samples per chain
    runner.ratios = [0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0]
    runner.dimensions = [2, 4]  # Can extend to [2, 4, 8] for more comprehensive results
    runner.basis_types = ["identity", "skewed", "ill_conditioned"]
    
    logger.info(f"Configuration:")
    logger.info(f"  Chains per config: {runner.num_chains}")
    logger.info(f"  Samples per chain: {runner.num_samples}")
    logger.info(f"  Dimensions: {runner.dimensions}")
    logger.info(f"  Basis types: {runner.basis_types}")
    logger.info(f"  σ/η ratios: {runner.ratios}")
    
    # Run the complete analysis
    try:
        runner.run_complete_analysis()
        logger.info("Experiments completed successfully!")
    except Exception as e:
        logger.error(f"Error during experiments: {e}")
        raise
    
    # Print summary of results
    logger.info("\nExperiment Summary:")
    logger.info(f"Results saved to: {runner.output_dir}")
    logger.info(f"Data files: {runner.data_dir}")
    logger.info(f"Plot files: {runner.plot_dir}")
    
    # Read and display key findings if available
    report_path = runner.output_dir / "experiment_report.json"
    if report_path.exists():
        import json
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        logger.info("\nKey Findings:")
        for finding in report.get('key_findings', []):
            logger.info(f"  {finding['basis_type']} {finding['dimension']}D: "
                       f"optimal ratio={finding['optimal_ratio']:.2f}, "
                       f"TV={finding['optimal_tv']:.6f}")


if __name__ == "__main__":
    main()