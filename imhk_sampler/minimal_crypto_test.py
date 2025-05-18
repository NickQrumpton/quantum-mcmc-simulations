"""
Minimal test for cryptographic lattice bases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import create_lattice_basis
from samplers import imhk_sampler_wrapper
from parameter_config import compute_smoothing_parameter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basis(dim, basis_type, sigma_ratio=2.0):
    """Test a single basis configuration."""
    logger.info(f"Testing {basis_type} basis with dimension {dim}")
    
    try:
        # Create the basis
        basis_info = create_lattice_basis(dim, basis_type)
        logger.info(f"Created {basis_type} basis")
        
        # Check what type of basis we got
        if isinstance(basis_info, tuple):
            logger.info(f"Got polynomial basis: degree={basis_info[0].degree()}, q={basis_info[1]}")
            # For polynomial bases, use a fixed sigma
            sigma = 10.0  # Reasonable value for cryptographic lattices
        else:
            # Compute smoothing parameter for matrix bases
            eta = compute_smoothing_parameter(basis_info)
            sigma = sigma_ratio * eta
            logger.info(f"Computed eta={eta}, sigma={sigma}")
        
        # Run sampler
        samples, metadata = imhk_sampler_wrapper(
            basis_info=basis_info,
            sigma=sigma,
            num_samples=10,  # Just 10 samples for testing
            burn_in=50,      # Small burn-in for testing
            basis_type=basis_type
        )
        
        logger.info(f"Sampling successful!")
        logger.info(f"Acceptance rate: {metadata.get('acceptance_rate', 'N/A')}")
        logger.info(f"Sample shape: {samples.shape}")
        
    except Exception as e:
        logger.error(f"Failed on {basis_type} with dimension {dim}: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Test each basis type with small dimensions
    for basis_type in ['q-ary', 'NTRU', 'PrimeCyclotomic']:
        test_basis(32, basis_type)
        print("-" * 50)

if __name__ == "__main__":
    main()