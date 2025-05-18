"""
Verify code quality and completeness for publication-ready results.
"""

import subprocess
import sys
from pathlib import Path
import json

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {description} - PASSED")
            return True
        else:
            print(f"✗ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {description} - ERROR: {e}")
        return False

def main():
    """Verify code quality for publication."""
    checks = []
    
    # 1. Import checks
    print("=== Import Verification ===")
    try:
        from experiments.report import ExperimentRunner
        from stats import compute_total_variation_distance
        from parameter_config import compute_smoothing_parameter
        from samplers import imhk_sampler, klein_sampler
        from utils import create_lattice_basis
        from diagnostics import compute_ess
        checks.append(("imports", True))
        print("✓ All core imports successful")
    except Exception as e:
        checks.append(("imports", False))
        print(f"✗ Import error: {e}")
    
    # 2. Basic functionality tests
    print("\n=== Basic Functionality ===")
    try:
        basis = create_lattice_basis(2, "identity")
        eta = compute_smoothing_parameter(basis)
        samples, metadata = imhk_sampler(B=basis, sigma=1.5*eta, num_samples=100)
        tv_dist = compute_total_variation_distance(samples, 1.5*eta, basis)
        
        if 0 <= tv_dist <= 1 and metadata['acceptance_rate'] > 0:
            checks.append(("basic_functionality", True))
            print(f"✓ Basic test: TV={tv_dist:.4f}, acceptance={metadata['acceptance_rate']:.3f}")
        else:
            checks.append(("basic_functionality", False))
            print("✗ Basic functionality test failed")
    except Exception as e:
        checks.append(("basic_functionality", False))
        print(f"✗ Basic functionality error: {e}")
    
    # 3. Extended dimension test
    print("\n=== Extended Dimension Test ===")
    try:
        for dim in [4, 8, 16]:
            basis = create_lattice_basis(dim, "identity")
            eta = compute_smoothing_parameter(basis)
            samples, metadata = imhk_sampler(B=basis, sigma=2*eta, num_samples=50)
            print(f"✓ Dimension {dim}: acceptance={metadata['acceptance_rate']:.3f}")
        checks.append(("dimensions", True))
    except Exception as e:
        checks.append(("dimensions", False))
        print(f"✗ Dimension test error: {e}")
    
    # 4. All basis types
    print("\n=== Basis Type Tests ===")
    try:
        for basis_type in ["identity", "skewed", "ill-conditioned"]:
            basis = create_lattice_basis(4, basis_type)
            eta = compute_smoothing_parameter(basis)
            samples, metadata = imhk_sampler(B=basis, sigma=2*eta, num_samples=50)
            print(f"✓ {basis_type}: acceptance={metadata['acceptance_rate']:.3f}")
        checks.append(("basis_types", True))
    except Exception as e:
        checks.append(("basis_types", False))
        print(f"✗ Basis type test error: {e}")
    
    # 5. Experiment runner verification
    print("\n=== Experiment Runner Test ===")
    try:
        runner = ExperimentRunner(output_dir="test_verification")
        runner.dimensions = [2]
        runner.basis_types = ["identity"]
        runner.ratios = [1.0]
        runner.num_chains = 1
        runner.num_samples = 100
        
        config = {
            'dimension': 2,
            'basis_type': 'identity',
            'ratio': 1.0
        }
        
        result = runner.run_single_experiment(config)
        if not np.isnan(result['tv_mean']):
            checks.append(("experiment_runner", True))
            print(f"✓ Experiment runner: TV={result['tv_mean']:.4f}")
        else:
            checks.append(("experiment_runner", False))
            print("✗ Experiment runner failed")
    except Exception as e:
        checks.append(("experiment_runner", False))
        print(f"✗ Experiment runner error: {e}")
    
    # Summary
    print("\n=== VERIFICATION SUMMARY ===")
    passed = sum(1 for _, status in checks if status)
    total = len(checks)
    
    for test, status in checks:
        print(f"{test}: {'PASSED' if status else 'FAILED'}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ Code is ready for publication-quality experiments!")
        return 0
    else:
        print("\n✗ Code needs fixes before running full experiments")
        return 1

if __name__ == "__main__":
    import numpy as np
    sys.exit(main())