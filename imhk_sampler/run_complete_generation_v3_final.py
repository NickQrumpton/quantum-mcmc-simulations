#!/usr/bin/env python
"""
Master script to run the complete V3 generation pipeline with all fixes applied.
"""

import os
import sys
import subprocess
import time
import logging
from datetime import datetime

# Set up logging
log_dir = "results/logs"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"complete_generation_final_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def print_separator(title):
    """Print a separator in logs."""
    separator = "=" * 50
    logger.info(f"\n{separator}")
    logger.info(f"{title}")
    logger.info(f"{separator}\n")

def run_command(command, description):
    """Run a command and log the results."""
    print_separator(description)
    logger.info(f"Running: {command}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        
        duration = time.time() - start_time
        logger.info(f"✓ Completed in {duration:.2f} seconds")
        
        if result.stdout:
            logger.info(f"STDOUT:\n{result.stdout}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logger.error(f"✗ Failed after {duration:.2f} seconds")
        logger.error(f"Exit code: {e.returncode}")
        
        if e.stdout:
            logger.error(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            logger.error(f"STDERR:\n{e.stderr}")
            
        return False

def main():
    """Run the complete V3 generation pipeline with fully fixed components."""
    logger.info("Starting Complete V3 Generation Pipeline (FINAL)")
    logger.info(f"Log file: {log_file}")
    
    # Track overall success
    all_success = True
    
    # Step 1: Clean environment
    if not run_command(
        "rm -rf results/data/* results/plots/*",
        "Cleaning Results Directory"
    ):
        all_success = False
    
    # Step 2: Run the fixed publication results generator
    if not run_command(
        f"sage fixed_publication_results_v3_final.py",
        "Running Publication Results Generator (FINAL)"
    ):
        all_success = False
    
    # Step 3: Test data integrity
    if not run_command(
        f"sage test_data_integrity_v3.py",
        "Testing Data Integrity"
    ):
        all_success = False
    
    # Step 4: Generate all results with patched components
    if not run_command(
        f"sage generate_all_results_patched.py",
        "Generating All Results (Patched)"
    ):
        all_success = False
    
    # Summary
    print_separator("Pipeline Summary")
    
    if all_success:
        logger.info("✓ All steps completed successfully")
    else:
        logger.error("✗ Some steps failed - check logs above")
    
    # List generated files
    logger.info("\nGenerated Files:")
    
    # Data files
    data_dir = "results/data"
    if os.path.exists(data_dir):
        for file in sorted(os.listdir(data_dir)):
            logger.info(f"  - {data_dir}/{file}")
    
    # Fixed publication results
    fixed_dir = "fixed_publication_results"
    if os.path.exists(fixed_dir):
        for root, dirs, files in os.walk(fixed_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), ".")
                logger.info(f"  - {rel_path}")
    
    # Check for critical files
    critical_files = [
        "fixed_publication_results/data/all_results.json",
        "fixed_publication_results/data/all_results.csv",
        "fixed_publication_results/figures/fig1_tv_distance_comparison.png",
        "fixed_publication_results/figures/fig2_acceptance_rates_heatmap.png",
        "fixed_publication_results/figures/fig3_performance_analysis.png",
        "fixed_publication_results/publication_report.json"
    ]
    
    logger.info("\nCritical Files Check:")
    for file in critical_files:
        if os.path.exists(file):
            logger.info(f"  ✓ {file}")
        else:
            logger.error(f"  ✗ {file} - MISSING")
    
    # Final status
    logger.info(f"\nPipeline completed. Check log at: {log_file}")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())