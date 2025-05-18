#!/usr/bin/env python3
"""
test_minimal.py - Minimal Test Suite for IMHK Sampler Framework

This script tests the core functionality of the Independent Metropolis-Hastings-Klein (IMHK)
Sampler framework to ensure it works correctly. It verifies:
1. SageMath functionality
2. Directory structure for results
3. Core samplers (klein_sampler and imhk_sampler)

Usage:
    python test_minimal.py

Author: Quantum MCMC Research Team
"""

import os
import sys
import traceback
import time
import subprocess
from pathlib import Path
import numpy as np
import importlib.util

# Create a simple step-by-step test framework
class TestStep:
    """Simple test step class to track and report test progress."""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.passed = False
        self.error = None
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start the test step and record time."""
        print(f"\n{'='*80}")
        print(f"TESTING: {self.name}")
        print(f"Description: {self.description}")
        print(f"{'-'*80}")
        self.start_time = time.time()
        
    def finish(self, success=True, error=None):
        """Finish the test step and record result."""
        self.end_time = time.time()
        self.passed = success
        self.error = error
        
        duration = self.end_time - self.start_time
        
        if success:
            print(f"\n✓ PASS: {self.name} ({duration:.2f}s)")
        else:
            print(f"\n✗ FAIL: {self.name} ({duration:.2f}s)")
            if error:
                print(f"\nError details:")
                print(f"{'-'*40}")
                print(f"{error}")
                if hasattr(error, '__traceback__'):
                    traceback.print_tb(error.__traceback__)
        
        return success


def check_sagemath_installation():
    """Check if SageMath is installed and provide installation instructions if needed."""
    # Check if sage module is available
    sage_spec = importlib.util.find_spec("sage")
    
    if sage_spec is None:
        print("\n" + "="*80)
        print("SageMath NOT FOUND - Installation Required")
        print("="*80)
        print("The IMHK Sampler requires SageMath for lattice operations.")
        print("\nInstallation Options:")
        print("1. Using conda (recommended):")
        print("   conda install -c conda-forge sagemath")
        print("\n2. Using pip (partial installation, may have limitations):")
        print("   pip install sagemath")
        print("\n3. For a full installation, visit:")
        print("   https://doc.sagemath.org/html/en/installation/index.html")
        print("\nAfter installing SageMath, run this test again.")
        print("="*80)
        return False
    
    return True

def test_sagemath_setup():
    """Test that SageMath is properly installed and configured."""
    test = TestStep(
        "SageMath Installation", 
        "Verify that SageMath is properly installed and can create matrices and vectors."
    )
    test.start()
    
    # First check if SageMath is installed
    if not check_sagemath_installation():
        return test.finish(False, ImportError("SageMath is not installed. Please install it using the instructions above."))
    
    try:
        # Import SageMath components
        from sage.all import matrix, vector, RR, ZZ, QQ
        
        # Create a simple matrix and vector
        dimension = 2
        M = matrix(RR, [[1, 0], [0, 1]])
        v = vector(RR, [1, 2])
        
        # Test matrix-vector multiplication
        result = M * v
        
        # Verify result
        expected = vector(RR, [1, 2])
        if not all(abs(result[i] - expected[i]) < 1e-10 for i in range(dimension)):
            raise ValueError(f"Matrix-vector multiplication failed: expected {expected}, got {result}")
            
        print(f"Created {dimension}x{dimension} matrix: {M}")
        print(f"Created vector: {v}")
        print(f"Matrix * vector = {result} ✓")
        
        return test.finish(True)
    
    except Exception as e:
        return test.finish(False, e)


def test_directory_setup():
    """Test and create the necessary directory structure for results."""
    test = TestStep(
        "Directory Structure",
        "Create and verify the necessary directory structure for results."
    )
    test.start()
    
    try:
        # Define required directories
        project_root = Path(__file__).resolve().parent
        required_dirs = [
            project_root / "results",
            project_root / "results/plots",
            project_root / "results/logs",
            project_root / "data"
        ]
        
        # Create directories if they don't exist
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            if not directory.exists():
                raise RuntimeError(f"Failed to create directory: {directory}")
            print(f"Directory verified: {directory} ✓")
        
        return test.finish(True)
    
    except Exception as e:
        return test.finish(False, e)


def test_klein_sampler():
    """Test the Klein sampler with a simple 2D lattice."""
    test = TestStep(
        "Klein Sampler",
        "Test the Klein sampler with a simple 2D lattice to verify it produces valid samples."
    )
    test.start()
    
    try:
        # Import the necessary functions
        from sage.all import matrix, vector, RR
        from imhk_sampler import klein_sampler_single
        
        # Create a 2D identity lattice basis
        dimension = 2
        B = matrix.identity(RR, dimension)
        print(f"Testing with {dimension}D identity lattice basis")
        
        # Parameters for the sampler
        sigma = 2.0
        center = vector(RR, [0, 0])
        num_samples = 10
        
        print(f"Generating {num_samples} samples with σ={sigma}")
        
        # Generate samples
        samples = []
        for i in range(num_samples):
            sample = klein_sampler_single(B, sigma, center)
            samples.append(sample)
            print(f"  Sample {i+1}: {sample}")
        
        # Verify samples have the correct dimension
        if any(len(sample) != dimension for sample in samples):
            raise ValueError("Some samples have incorrect dimension")
        
        # Verify samples are lattice points (integers since basis is identity)
        from sage.rings.integer import Integer
        integer_samples = all(all(isinstance(x, Integer) or x.is_integer() for x in sample) 
                             for sample in samples)
        if not integer_samples:
            raise ValueError("Some samples are not lattice points (should be integers for identity basis)")
        
        # Calculate mean and variance as a basic check
        mean_x = sum(sample[0] for sample in samples) / num_samples
        mean_y = sum(sample[1] for sample in samples) / num_samples
        var_x = sum((sample[0] - mean_x)**2 for sample in samples) / num_samples
        var_y = sum((sample[1] - mean_y)**2 for sample in samples) / num_samples
        
        print(f"Sample statistics:")
        print(f"  Mean: ({mean_x:.2f}, {mean_y:.2f})")
        print(f"  Variance: ({var_x:.2f}, {var_y:.2f})")
        print(f"  Expected variance (approximately): σ² = {sigma**2}")
        
        # Basic check: variance should be approximately sigma^2 for standard cases
        # This is a rough check, not precise for small sample sizes
        variance_tolerance = 1.5  # Allow some variation due to small sample size
        if var_x > sigma**2 * variance_tolerance or var_y > sigma**2 * variance_tolerance:
            print(f"Warning: Sample variance exceeds expected variance by more than {variance_tolerance}x")
            print(f"This might be normal with a small sample size ({num_samples})")
        
        return test.finish(True)
    
    except Exception as e:
        return test.finish(False, e)


def test_imhk_sampler():
    """Test the IMHK sampler with a small number of samples."""
    test = TestStep(
        "IMHK Sampler",
        "Test the Independent Metropolis-Hastings-Klein sampler with a small number of samples."
    )
    test.start()
    
    try:
        # Import the necessary functions
        from sage.all import matrix, vector, RR
        from imhk_sampler import imhk_sampler
        
        # Create a slightly skewed lattice basis for more interesting test
        dimension = 2
        B = matrix(RR, [[1, 0.5], [0, 1]])
        print(f"Testing with {dimension}D skewed lattice basis:")
        print(B)
        
        # Parameters for the sampler
        sigma = 2.0
        center = vector(RR, [0, 0])
        num_samples = 10
        burn_in = 5
        
        print(f"Generating {num_samples} samples with σ={sigma}, burn-in={burn_in}")
        
        # Generate samples
        start_time = time.time()
        samples, acceptance_rate, all_samples, all_accepts = imhk_sampler(
            B, sigma, num_samples, center, burn_in=burn_in
        )
        elapsed_time = time.time() - start_time
        
        # Print basic information
        print(f"Sampling completed in {elapsed_time:.2f} seconds")
        print(f"Acceptance rate: {acceptance_rate:.4f}")
        
        # Display samples
        print("Final samples:")
        for i, sample in enumerate(samples):
            print(f"  Sample {i+1}: {sample}")
        
        # Verify we have the expected number of samples
        if len(samples) != num_samples:
            raise ValueError(f"Expected {num_samples} samples, got {len(samples)}")
        
        # Verify samples have the correct dimension
        if any(len(sample) != dimension for sample in samples):
            raise ValueError("Some samples have incorrect dimension")
        
        # Verify samples are lattice points (should satisfy x = a + 0.5*b and y = b for integers a,b)
        is_lattice_point = lambda v: (v[1].is_integer() and 
                                     (v[0] - 0.5*v[1]).is_integer())
        if not all(is_lattice_point(sample) for sample in samples):
            raise ValueError("Some samples are not lattice points for the given basis")
        
        # Calculate mean and variance as a basic check
        mean_x = sum(sample[0] for sample in samples) / num_samples
        mean_y = sum(sample[1] for sample in samples) / num_samples
        var_x = sum((sample[0] - mean_x)**2 for sample in samples) / num_samples
        var_y = sum((sample[1] - mean_y)**2 for sample in samples) / num_samples
        
        print(f"Sample statistics:")
        print(f"  Mean: ({mean_x:.2f}, {mean_y:.2f})")
        print(f"  Variance: ({var_x:.2f}, {var_y:.2f})")
        
        # Check if all_samples and all_accepts have the expected length
        expected_all_length = num_samples + burn_in
        if len(all_samples) != expected_all_length:
            raise ValueError(f"Expected {expected_all_length} elements in all_samples, got {len(all_samples)}")
        if len(all_accepts) != expected_all_length:
            raise ValueError(f"Expected {expected_all_length} elements in all_accepts, got {len(all_accepts)}")
        
        # Verify the burn-in samples aren't included in the main samples
        if any(any(all_samples[i][j] == samples[0][j] for j in range(dimension)) for i in range(burn_in)):
            print("Warning: Possible overlap between burn-in samples and returned samples")
        
        return test.finish(True)
    
    except Exception as e:
        return test.finish(False, e)


def test_full_pipeline():
    """Test a simple end-to-end pipeline using both samplers."""
    test = TestStep(
        "Full Pipeline",
        "Test a simple end-to-end pipeline using both samplers and basic visualization."
    )
    test.start()
    
    try:
        # Import necessary functions
        from sage.all import matrix, vector, RR
        from imhk_sampler import klein_sampler, imhk_sampler, plot_2d_samples
        
        # Create a simple lattice basis
        dimension = 2
        B = matrix.identity(RR, dimension)
        print(f"Testing full pipeline with {dimension}D identity lattice basis")
        
        # Parameters
        sigma = 3.0
        center = vector(RR, [0, 0])
        klein_samples_count = 100
        imhk_samples_count = 100
        burn_in = 50
        
        print(f"Generating samples with σ={sigma}")
        
        # Generate samples with Klein sampler
        print(f"Klein sampler: Generating {klein_samples_count} samples...")
        klein_samples = klein_sampler(B, sigma, klein_samples_count, center)
        print(f"Klein sampling completed.")
        
        # Generate samples with IMHK sampler
        print(f"IMHK sampler: Generating {imhk_samples_count} samples with burn-in={burn_in}...")
        imhk_results = imhk_sampler(B, sigma, imhk_samples_count, center, burn_in=burn_in)
        imhk_samples = imhk_results[0]  # Extract just the samples
        acceptance_rate = imhk_results[1]
        print(f"IMHK sampling completed. Acceptance rate: {acceptance_rate:.4f}")
        
        # Test visualization by creating a basic plot
        print(f"Creating visualization plots...")
        
        # Save plots to results/plots directory
        plot_2d_samples(
            klein_samples, 
            sigma, 
            "test_klein_samples.png", 
            B, 
            "Klein Sampler Test", 
            center
        )
        
        plot_2d_samples(
            imhk_samples, 
            sigma, 
            "test_imhk_samples.png", 
            B, 
            "IMHK Sampler Test", 
            center
        )
        
        print(f"Visualization completed. Plots saved to results/plots/")
        
        return test.finish(True)
    
    except Exception as e:
        return test.finish(False, e)


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("IMHK SAMPLER MINIMAL TEST SUITE")
    print("="*80)
    
    # Add initialization message
    print("\nInitializing test environment...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check if SageMath is installed first
    sage_available = check_sagemath_installation()
    
    # List of all tests
    all_tests = [
        test_sagemath_setup,
        test_directory_setup,
        test_klein_sampler,
        test_imhk_sampler,
        test_full_pipeline
    ]
    
    # Filter tests based on SageMath availability
    if not sage_available:
        print("\nSKIPPING SageMath-dependent tests due to missing SageMath installation.")
        # Only run directory setup test which doesn't depend on SageMath
        tests = [test_directory_setup]
    else:
        tests = all_tests
    
    # Run tests and collect results
    results = []
    for test_func in tests:
        result = test_func()
        results.append((test_func.__name__, result))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    tests_run = 0
    for name, passed in results:
        tests_run += 1
        if passed:
            print(f"✓ PASS: {name}")
        else:
            print(f"✗ FAIL: {name}")
            all_passed = False
    
    # Print skipped tests if any
    if not sage_available:
        skipped_tests = [t.__name__ for t in all_tests if t != test_directory_setup]
        for name in skipped_tests:
            print(f"⚠ SKIP: {name} (requires SageMath)")
    
    # Final verdict
    print("\n" + "="*80)
    if not sage_available:
        print("\n⚠️ PARTIAL TEST RUN - SAGEMATH REQUIRED FOR FULL TESTING ⚠️")
        print("\nPlease install SageMath and run the tests again for a complete evaluation.")
    elif all_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nThe IMHK Sampler framework is functioning correctly!")
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print("\nPlease review the test output for details on the failures.")
    
    print("\n" + "="*80)
    
    return all_passed and sage_available  # Return true only if all tests passed and SageMath is available


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)