"""
Simple test for cryptographic lattice bases.
"""

import sys
from pathlib import Path
import numpy as np
from math import sqrt

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import create_lattice_basis
from samplers import imhk_sampler_wrapper
from parameter_config import compute_smoothing_parameter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basis_type(basis_type, dim=32, num_samples=100):
    """Test a single basis type."""
    logger.info(f"\nTesting {basis_type} basis, dimension {dim}")
    
    try:
        # Create the basis
        basis_info = create_lattice_basis(dim, basis_type)
        
        # Determine appropriate sigma
        if isinstance(basis_info, tuple):
            # Structured lattice
            poly_mod, q = basis_info
            sigma = float(sqrt(q) / 20)  # Conservative value
            logger.info(f"Structured lattice: degree={poly_mod.degree()}, q={q}, sigma={sigma}")
        else:
            # Matrix lattice
            eta = compute_smoothing_parameter(basis_info)
            sigma = max(2.0 * eta, 1.0)  # Ensure reasonable minimum
            logger.info(f"Matrix lattice: eta={eta}, sigma={sigma}")
        
        # Run IMHK sampler
        samples, metadata = imhk_sampler_wrapper(
            basis_info=basis_info,
            sigma=sigma,
            num_samples=num_samples,
            burn_in=50,  # Small burn-in for testing
            basis_type=basis_type
        )
        
        logger.info(f"Success! Acceptance rate: {metadata.get('acceptance_rate', 'N/A'):.4f}")
        logger.info(f"Sample shape: {samples.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple tests for all basis types."""
    basis_types = ['identity', 'q-ary', 'NTRU', 'PrimeCyclotomic']
    
    logger.info("Starting cryptographic lattice tests")
    results = {}
    
    for basis_type in basis_types:
        success = test_basis_type(basis_type, dim=16, num_samples=50)
        results[basis_type] = success
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    for basis_type, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{basis_type:15} {status}")
    
    all_passed = all(results.values())
    logger.info(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

if __name__ == "__main__":
    main()