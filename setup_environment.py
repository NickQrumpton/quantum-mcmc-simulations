#!/usr/bin/env python3
"""
setup_environment.py - Environment setup for IMHK Sampler

This script helps set up the necessary environment for running the IMHK Sampler,
including checking for and installing required dependencies.

Key features:
1. Checks for required Python packages
2. Verifies SageMath installation
3. Provides installation instructions for missing components
4. Creates necessary directories for results

Usage:
    python setup_environment.py [--install] [--force-reinstall] [--no-sage]
    
Options:
    --install          Attempt to install missing packages
    --force-reinstall  Force reinstall of all packages
    --no-sage          Skip SageMath check (for environments without SageMath)

Author: Quantum MCMC Research Team
"""

import os
import sys
import subprocess
import platform
import importlib.util
import argparse
from pathlib import Path

# Required packages
REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
    "tqdm"
]

# SageMath is special and handled differently
SAGE_REQUIRED = True

# Colors for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_colored(text, color):
    """Print colored text."""
    print(f"{color}{text}{Colors.ENDC}")

def print_header(text):
    """Print a header with formatting."""
    print("\n" + "="*80)
    print_colored(text, Colors.BOLD + Colors.BLUE)
    print("="*80)

def print_subheader(text):
    """Print a subheader with formatting."""
    print("\n" + "-"*40)
    print_colored(text, Colors.BOLD)
    print("-"*40)

def check_package(package_name):
    """Check if a package is installed and can be imported."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name, force=False):
    """Install a package using pip."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if force:
        cmd.append("--force-reinstall")
    cmd.append(package_name)
    
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        print_colored(f"Failed to install {package_name}", Colors.RED)
        return False

def check_sagemath():
    """Check if SageMath is installed and can be imported."""
    sage_spec = importlib.util.find_spec("sage")
    if sage_spec is None:
        print_colored("\nSageMath not found. This is required for the IMHK Sampler.", Colors.YELLOW)
        print("\nInstallation Options:")
        print_colored("1. Using conda (recommended):", Colors.BOLD)
        print("   conda install -c conda-forge sagemath")
        
        print_colored("\n2. Using pip (partial installation, may have limitations):", Colors.BOLD)
        print("   pip install sagemath")
        
        print_colored("\n3. For a full installation, visit:", Colors.BOLD)
        print("   https://doc.sagemath.org/html/en/installation/index.html")
        
        return False
    
    print_colored("SageMath is installed ✓", Colors.GREEN)
    return True

def create_directories():
    """Create necessary directories for results."""
    dirs = [
        Path('results'),
        Path('results/plots'),
        Path('results/logs'),
        Path('data')
    ]
    
    all_created = True
    for directory in dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Directory verified: {directory} ✓")
        except Exception as e:
            print_colored(f"Failed to create directory {directory}: {e}", Colors.RED)
            all_created = False
    
    return all_created

def main():
    """Main function to check and set up the environment."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Set up environment for IMHK Sampler")
    parser.add_argument("--install", action="store_true", help="Attempt to install missing packages")
    parser.add_argument("--force-reinstall", action="store_true", help="Force reinstall of all packages")
    parser.add_argument("--no-sage", action="store_true", help="Skip SageMath check")
    args = parser.parse_args()
    
    print_header("IMHK Sampler Environment Setup")
    
    # System information
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check required packages
    print_subheader("Checking Required Packages")
    
    all_packages_installed = True
    missing_packages = []
    
    for package in REQUIRED_PACKAGES:
        if check_package(package):
            print_colored(f"{package} is installed ✓", Colors.GREEN)
        else:
            print_colored(f"{package} is NOT installed ✗", Colors.RED)
            all_packages_installed = False
            missing_packages.append(package)
    
    # Install missing packages if requested
    if not all_packages_installed and args.install:
        print_subheader("Installing Missing Packages")
        for package in missing_packages:
            success = install_package(package, args.force_reinstall)
            if not success:
                all_packages_installed = False
    elif not all_packages_installed:
        print_colored("\nSome required packages are missing. Run with --install to install them.", Colors.YELLOW)
    
    # Check SageMath
    if SAGE_REQUIRED and not args.no_sage:
        print_subheader("Checking SageMath")
        sage_installed = check_sagemath()
    else:
        sage_installed = not SAGE_REQUIRED or args.no_sage
        if args.no_sage:
            print_colored("\nSagemath check skipped as requested.", Colors.YELLOW)
            print("Note: SageMath is required for full functionality of the IMHK Sampler.")
    
    # Create directories
    print_subheader("Setting Up Directories")
    dirs_created = create_directories()
    
    # Overall status
    print_subheader("Setup Summary")
    
    if all_packages_installed and (sage_installed or args.no_sage) and dirs_created:
        print_colored("✓ All checks passed! Your environment is ready to use.", Colors.GREEN)
        return 0
    else:
        print_colored("⚠ Some components are missing or could not be set up.", Colors.YELLOW)
        print("Please address the issues above before running the IMHK Sampler.")
        
        if not sage_installed and not args.no_sage:
            print_colored("\nNote: SageMath is optional but highly recommended.", Colors.YELLOW)
            print("You can run with the --no-sage flag to skip the SageMath check.")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())