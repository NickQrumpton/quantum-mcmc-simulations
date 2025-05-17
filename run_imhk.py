#!/usr/bin/env python3
"""
run_imhk.py - Main entry point for running IMHK Sampler experiments

This script provides a convenient way to run different experiments and demos
with the Independent Metropolis-Hastings-Klein (IMHK) sampler. It checks for
required dependencies and provides a command-line interface.

Usage:
    python run_imhk.py [experiment_type]
    
Experiment types:
    basic       Run a basic 2D example (default)
    sweep       Run a parameter sweep (dimensions, sigmas, basis types)
    test        Run the minimal test suite
    validate    Verify the framework is functional
    
Examples:
    python run_imhk.py basic
    python run_imhk.py test

Author: Quantum MCMC Research Team
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path
import subprocess

# Check if SageMath is installed
def check_sagemath():
    """Check if SageMath is installed properly."""
    sage_spec = importlib.util.find_spec("sage")
    if sage_spec is None:
        print("\n" + "="*80)
        print("ERROR: SageMath NOT FOUND")
        print("="*80)
        print("The IMHK Sampler requires SageMath for lattice operations.")
        print("\nInstallation Options:")
        print("1. Using conda (recommended):")
        print("   conda install -c conda-forge sagemath")
        print("\n2. Using pip (partial installation, may have limitations):")
        print("   pip install sagemath")
        print("\n3. For a full installation, visit:")
        print("   https://doc.sagemath.org/html/en/installation/index.html")
        print("\nAfter installing SageMath, run this script again.")
        print("="*80)
        return False
    return True

# Check for required dependencies
def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        "numpy", 
        "scipy", 
        "matplotlib", 
        "sage"
    ]
    
    missing = []
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
    
    if missing:
        print("\n" + "="*80)
        print(f"ERROR: Missing required packages: {', '.join(missing)}")
        print("="*80)
        print("Please install the missing packages before running this script.")
        print("You can run the setup script to install dependencies:")
        print("  python setup_environment.py --install")
        print("="*80)
        return False
    
    return True

# Run the setup environment script
def run_setup(install=False):
    """Run the setup environment script."""
    setup_script = Path("setup_environment.py")
    
    if not setup_script.exists():
        print("ERROR: setup_environment.py not found.")
        return False
    
    cmd = [sys.executable, str(setup_script)]
    if install:
        cmd.append("--install")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print("ERROR: Failed to run setup script.")
        return False

# Run the basic example
def run_basic_example():
    """Run a basic 2D example with the IMHK sampler."""
    try:
        from imhk_sampler.main import run_basic_example
        run_basic_example()
        return True
    except ImportError as e:
        print(f"ERROR: Failed to import IMHK sampler: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Failed to run basic example: {e}")
        return False

# Run parameter sweep
def run_parameter_sweep():
    """Run a comprehensive parameter sweep."""
    try:
        from imhk_sampler.main import run_comprehensive_tests
        # Use limited parameters for faster execution
        dimensions = [2, 3]
        sigmas = [1.0, 2.0, 3.0]
        basis_types = ['identity', 'skewed', 'ill-conditioned']
        run_comprehensive_tests(dimensions, sigmas, basis_types, num_samples=1000)
        return True
    except ImportError as e:
        print(f"ERROR: Failed to import IMHK sampler: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Failed to run parameter sweep: {e}")
        return False

# Run the minimal test suite
def run_tests():
    """Run the minimal test suite."""
    test_script = Path("imhk_sampler/test_minimal.py")
    
    if not test_script.exists():
        print("ERROR: test_minimal.py not found.")
        return False
    
    try:
        subprocess.run([sys.executable, str(test_script)], check=True)
        return True
    except subprocess.CalledProcessError:
        print("ERROR: Tests failed.")
        return False

# Validate the framework functionality
def validate_framework():
    """Validate the framework functionality."""
    try:
        # Import key components
        from imhk_sampler import (
            imhk_sampler, klein_sampler, 
            compute_autocorrelation, compute_total_variation_distance
        )
        from sage.all import matrix, RR
        
        # Create a simple lattice basis
        dim = 2
        B = matrix.identity(RR, dim)
        sigma = 2.0
        
        # Generate a few samples
        print("Generating samples with Klein's algorithm...")
        klein_sample = klein_sampler(B, sigma)
        
        print("Generating samples with IMHK sampler...")
        samples, acceptance_rate, _, _ = imhk_sampler(
            B, sigma, num_samples=10, burn_in=5
        )
        
        print(f"Results:")
        print(f"  Klein sample: {klein_sample}")
        print(f"  IMHK acceptance rate: {acceptance_rate:.4f}")
        print(f"  First IMHK sample: {samples[0]}")
        
        print("\nFramework validation successful!")
        return True
    except Exception as e:
        print(f"ERROR: Framework validation failed: {e}")
        return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run IMHK Sampler experiments"
    )
    parser.add_argument(
        "experiment", 
        choices=["basic", "sweep", "test", "validate"], 
        default="basic",
        nargs="?",
        help="Type of experiment to run"
    )
    parser.add_argument(
        "--setup", 
        action="store_true",
        help="Run setup environment script first"
    )
    parser.add_argument(
        "--install", 
        action="store_true",
        help="Install missing dependencies"
    )
    
    args = parser.parse_args()
    
    # Run setup if requested
    if args.setup:
        success = run_setup(args.install)
        if not success:
            return 1
    
    # Check dependencies
    if not check_dependencies() or not check_sagemath():
        print("\nWould you like to run the setup script? (y/n)")
        response = input().strip().lower()
        if response == 'y':
            success = run_setup(install=True)
            if not success:
                return 1
        else:
            return 1
    
    # Run the requested experiment
    if args.experiment == "basic":
        success = run_basic_example()
    elif args.experiment == "sweep":
        success = run_parameter_sweep()
    elif args.experiment == "test":
        success = run_tests()
    elif args.experiment == "validate":
        success = validate_framework()
    else:
        print(f"Unknown experiment type: {args.experiment}")
        success = False
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())