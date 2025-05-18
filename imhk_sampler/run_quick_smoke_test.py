#!/usr/bin/env sage -python
"""
Quick smoke test to ensure basic functionality works.
This is a minimal test to verify the installation and basic operations.
"""

import sys
from pathlib import Path
import numpy as np
import logging
import time

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure minimal logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_lattice(dim=8, num_samples=50):
    """Test basic functionality with a simple identity lattice."""
    logger.info(f"Testing dimension {dim} with {num_samples} samples")
    
    try:
        # Import required modules
        from utils import create_lattice_basis
        from samplers import imhk_sampler_wrapper
        from parameter_config import compute_smoothing_parameter
        
        # Create simple identity lattice
        logger.info("Creating identity lattice...")
        basis = create_lattice_basis(dim, 'identity')
        
        # Compute sigma
        eta = compute_smoothing_parameter(basis)
        sigma = 2.0 * eta
        logger.info(f"Computed sigma: {sigma}")
        
        # Run IMHK sampler
        logger.info("Running IMHK sampler...")
        start_time = time.time()
        
        samples, metadata = imhk_sampler_wrapper(
            basis_info=basis,
            sigma=sigma,
            num_samples=num_samples,
            burn_in=20,
            basis_type='identity'
        )
        
        elapsed = time.time() - start_time
        
        logger.info(f"Sampling completed in {elapsed:.2f}s")
        logger.info(f"Generated {samples.shape[0]} samples")
        logger.info(f"Acceptance rate: {metadata.get('acceptance_rate', 0):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cryptographic_lattice():
    """Test cryptographic lattice functionality."""
    logger.info("\nTesting cryptographic lattice (q-ary)")
    
    try:
        from utils import create_lattice_basis
        from samplers import imhk_sampler_wrapper
        
        # Create q-ary lattice
        basis = create_lattice_basis(8, 'q-ary')
        
        # Run sampler with fixed sigma
        samples, metadata = imhk_sampler_wrapper(
            basis_info=basis,
            sigma=1.0,
            num_samples=20,
            burn_in=10,
            basis_type='q-ary'
        )
        
        logger.info(f"q-ary test completed successfully")
        logger.info(f"Acceptance rate: {metadata.get('acceptance_rate', 0):.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Cryptographic test failed: {e}")
        return False

def test_tv_distance():
    """Test TV distance computation (without optimization)."""
    logger.info("\nTesting TV distance computation")
    
    try:
        from utils import create_lattice_basis
        from stats import tv_distance_discrete_gaussian
        import numpy as np
        
        # Small test case
        dim = 4
        basis = create_lattice_basis(dim, 'identity')
        sigma = 1.0
        
        # Generate simple samples
        samples = np.random.normal(0, sigma, (20, dim))
        samples = np.round(samples).astype(int)
        
        # Compute TV distance with small radius
        logger.info("Computing TV distance...")
        tv_dist = tv_distance_discrete_gaussian(basis, sigma, samples, max_radius=2)
        
        logger.info(f"TV distance: {tv_dist:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"TV distance test failed: {e}")
        return False

def main():
    """Run quick smoke tests."""
    logger.info("IMHK Quick Smoke Test")
    logger.info("="*50)
    
    tests = [
        ("Basic Identity Lattice", lambda: test_basic_lattice(8, 50)),
        ("Cryptographic Lattice", test_cryptographic_lattice),
        ("TV Distance", test_tv_distance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning: {test_name}")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name:25} {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        logger.info("\nAll tests passed! The IMHK sampler is working correctly.")
        logger.info("\nNext steps:")
        logger.info("1. Run larger smoke test: sage run_smoke_test_crypto.py")
        logger.info("2. Run optimized test: sage run_optimized_smoke_test.py")
    else:
        logger.error("\nSome tests failed. Please check the error messages above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())