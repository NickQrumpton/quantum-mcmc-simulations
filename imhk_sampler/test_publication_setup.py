"""
Quick test to verify publication setup is working correctly.
"""

import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Test imports
    from utils import create_lattice_basis
    from samplers import imhk_sampler, klein_sampler
    from stats import compute_total_variation_distance
    from parameter_config import compute_smoothing_parameter
    from diagnostics import compute_ess
    
    logger.info("✓ All imports successful")
    
    # Test basic functionality
    basis = create_lattice_basis(2, "identity")
    logger.info("✓ Created lattice basis")
    
    eta = compute_smoothing_parameter(basis)
    sigma = 1.5 * eta
    logger.info(f"✓ Computed smoothing parameter: η={eta:.4f}, σ={sigma:.4f}")
    
    # Test IMHK sampler
    samples, metadata = imhk_sampler(
        B=basis,
        sigma=sigma,
        num_samples=100,
        burn_in=50
    )
    logger.info(f"✓ IMHK sampler: shape={samples.shape}, acceptance={metadata['acceptance_rate']:.3f}")
    
    # Test Klein sampler
    klein_samples = klein_sampler(
        B=basis,
        sigma=sigma,
        num_samples=100
    )
    logger.info(f"✓ Klein sampler: shape={klein_samples.shape}")
    
    # Test TV distance
    tv_dist = compute_total_variation_distance(samples, sigma, basis)
    logger.info(f"✓ TV distance: {tv_dist:.6f}")
    
    # Test ESS
    ess = compute_ess(samples)
    logger.info(f"✓ ESS: mean={np.mean(ess):.1f}")
    
    logger.info("\n✓ All tests passed! Publication setup is working correctly.")
    
except Exception as e:
    logger.error(f"✗ Test failed: {e}")
    raise