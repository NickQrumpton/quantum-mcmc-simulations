#!/usr/bin/env sage -python
"""
Debug script to identify specific errors in the crypto implementation.
"""

import sys
from pathlib import Path
import logging

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all required imports."""
    logger.info("Testing imports...")
    
    imports_to_test = [
        ("utils", ["create_lattice_basis"]),
        ("samplers", ["imhk_sampler_wrapper", "klein_sampler_wrapper"]),
        ("stats", ["compute_total_variation_distance"]),
        ("parameter_config", ["compute_smoothing_parameter"]),
        ("diagnostics", ["compute_ess"]),
    ]
    
    for module_name, functions in imports_to_test:
        try:
            module = __import__(module_name)
            for func in functions:
                if hasattr(module, func):
                    logger.info(f"✓ {module_name}.{func} imported successfully")
                else:
                    logger.error(f"✗ {module_name}.{func} not found")
        except Exception as e:
            logger.error(f"✗ Failed to import {module_name}: {e}")

def test_basis_creation():
    """Test creation of each basis type."""
    logger.info("\nTesting basis creation...")
    
    from utils import create_lattice_basis
    
    basis_types = ['identity', 'q-ary', 'NTRU', 'PrimeCyclotomic']
    dim = 8
    
    for basis_type in basis_types:
        try:
            logger.info(f"Creating {basis_type} basis...")
            basis = create_lattice_basis(dim, basis_type)
            
            if isinstance(basis, tuple):
                poly_mod, q = basis
                logger.info(f"✓ {basis_type}: polynomial degree={poly_mod.degree()}, q={q}")
            else:
                logger.info(f"✓ {basis_type}: matrix shape={basis.nrows()}x{basis.ncols()}")
        except Exception as e:
            logger.error(f"✗ {basis_type} failed: {e}")
            import traceback
            traceback.print_exc()

def test_sampler():
    """Test the sampler wrapper."""
    logger.info("\nTesting sampler...")
    
    try:
        from utils import create_lattice_basis
        from samplers import imhk_sampler_wrapper
        
        # Test with identity basis
        basis = create_lattice_basis(4, 'identity')
        sigma = 1.0
        
        logger.info("Running IMHK sampler...")
        samples, metadata = imhk_sampler_wrapper(
            basis_info=basis,
            sigma=sigma,
            num_samples=10,
            burn_in=5,
            basis_type='identity'
        )
        
        logger.info(f"✓ Sampler successful: {samples.shape[0]} samples generated")
        logger.info(f"  Acceptance rate: {metadata.get('acceptance_rate', 0):.4f}")
        
    except Exception as e:
        logger.error(f"✗ Sampler failed: {e}")
        import traceback
        traceback.print_exc()

def test_specific_error():
    """Test specific configuration that might be causing errors."""
    logger.info("\nTesting problematic configuration...")
    
    try:
        from utils import create_lattice_basis
        from samplers import imhk_sampler_wrapper
        from parameter_config import compute_smoothing_parameter
        
        # Test q-ary with dimension 32 (problematic case)
        dim = 32
        basis_type = 'q-ary'
        
        logger.info(f"Creating {basis_type} basis with dimension {dim}...")
        basis = create_lattice_basis(dim, basis_type)
        
        logger.info("Computing smoothing parameter...")
        eta = compute_smoothing_parameter(basis)
        sigma = 2.0 * eta
        
        logger.info(f"eta={eta}, sigma={sigma}")
        
        # Check if sigma is too small
        if sigma < 1e-6:
            logger.warning(f"Sigma is very small: {sigma}")
            sigma = max(sigma, 0.1)  # Use minimum sigma
            logger.info(f"Adjusted sigma to: {sigma}")
        
        logger.info("Running sampler...")
        samples, metadata = imhk_sampler_wrapper(
            basis_info=basis,
            sigma=sigma,
            num_samples=10,
            burn_in=5,
            basis_type=basis_type
        )
        
        logger.info(f"✓ Test successful")
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all debug tests."""
    logger.info("IMHK Crypto Debug Test")
    logger.info("="*50)
    
    tests = [
        ("Import Test", test_imports),
        ("Basis Creation", test_basis_creation),
        ("Sampler Test", test_sampler),
        ("Specific Error Test", test_specific_error),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{test_name}")
        logger.info("-"*30)
        try:
            test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\nDebug test complete. Check errors above.")

if __name__ == "__main__":
    main()