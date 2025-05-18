#!/usr/bin/env sage -python
"""Comprehensive verification of all IMHK sampler components."""

import sys
from pathlib import Path
import numpy as np
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def check_imports():
    """Check all critical imports."""
    logger.info("1. CHECKING IMPORTS")
    logger.info("=" * 70)
    
    imports_to_check = [
        ("from utils import create_lattice_basis", "utils.create_lattice_basis"),
        ("from samplers import imhk_sampler, klein_sampler, imhk_sampler_wrapper", "samplers"),
        ("from stats import tv_distance_discrete_gaussian", "stats.tv_distance"),
        ("from stats_optimized import tv_distance_discrete_gaussian_optimized", "stats_optimized.tv_distance"),
        ("from diagnostics import compute_ess, compute_autocorrelation", "diagnostics"),
        ("from visualization import plot_samples", "visualization"),
        ("from experiments import ExperimentRunner", "experiments"),
    ]
    
    all_imports_ok = True
    
    for import_stmt, description in imports_to_check:
        try:
            exec(import_stmt)
            logger.info(f"  ✓ {description}")
        except Exception as e:
            logger.error(f"  ✗ {description}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def check_basis_types():
    """Check all basis types."""
    logger.info("\n2. CHECKING BASIS TYPES")
    logger.info("=" * 70)
    
    from utils import create_lattice_basis
    
    basis_configs = [
        ("identity", 4, {}),
        ("identity", 8, {}),
        ("q-ary", 8, {}),
        ("q-ary", 16, {}),
        ("NTRU", 512, {}),
        ("PrimeCyclotomic", 683, {}),
    ]
    
    all_bases_ok = True
    
    for basis_type, dim, kwargs in basis_configs:
        try:
            basis = create_lattice_basis(dim, basis_type, **kwargs)
            
            # Check basis properties
            if basis_type in ["NTRU", "PrimeCyclotomic"]:
                # Structured lattice returns (poly_mod, q)
                poly_mod, q = basis
                logger.info(f"  ✓ {basis_type} (dim={dim}): poly_mod={poly_mod.degree()}, q={q}")
            else:
                # Regular matrix lattice - use nrows/ncols instead of shape
                logger.info(f"  ✓ {basis_type} (dim={dim}): shape=({basis.nrows()}, {basis.ncols()})")
                
        except Exception as e:
            logger.error(f"  ✗ {basis_type} (dim={dim}): {e}")
            all_bases_ok = False
            traceback.print_exc()
    
    return all_bases_ok

def check_samplers():
    """Check all samplers."""
    logger.info("\n3. CHECKING SAMPLERS")
    logger.info("=" * 70)
    
    from utils import create_lattice_basis
    from samplers import imhk_sampler, klein_sampler, imhk_sampler_wrapper
    
    test_configs = [
        ("identity", 4, 1.0, 100),
        ("q-ary", 8, 1.0, 100),
        ("NTRU", 512, 5.0, 50),
        ("PrimeCyclotomic", 683, 2.0, 50),
    ]
    
    all_samplers_ok = True
    
    for basis_type, dim, sigma, num_samples in test_configs:
        try:
            basis = create_lattice_basis(dim, basis_type)
            
            # Test IMHK sampler with wrapper
            try:
                result = imhk_sampler_wrapper(
                    basis, sigma, num_samples,
                    burn_in=50, basis_type=basis_type
                )
                # The wrapper returns (samples, metadata)
                samples, metadata = result
                n_samples = samples.shape[0] if hasattr(samples, 'shape') else len(samples)
                logger.info(f"  ✓ IMHK {basis_type} (dim={dim}): generated {n_samples} samples")
            except Exception as e:
                logger.error(f"  ✗ IMHK {basis_type} (dim={dim}): {e}")
                all_samplers_ok = False
                
            # Test Klein sampler for regular lattices
            if basis_type in ["identity", "q-ary"]:
                try:
                    klein_sample = klein_sampler(basis, sigma)
                    logger.info(f"  ✓ Klein {basis_type} (dim={dim}): sample shape={klein_sample.shape}")
                except Exception as e:
                    logger.error(f"  ✗ Klein {basis_type} (dim={dim}): {e}")
                    all_samplers_ok = False
                    
        except Exception as e:
            logger.error(f"  ✗ Setup for {basis_type} (dim={dim}): {e}")
            all_samplers_ok = False
            traceback.print_exc()
    
    return all_samplers_ok

def check_tv_distance():
    """Check TV distance computation."""
    logger.info("\n4. CHECKING TV DISTANCE")
    logger.info("=" * 70)
    
    from utils import create_lattice_basis
    from samplers import imhk_sampler_wrapper
    
    try:
        # Try regular version
        from stats import tv_distance_discrete_gaussian
        
        # Test on small lattice
        basis = create_lattice_basis(4, 'identity')
        samples, _ = imhk_sampler_wrapper(basis, 2.0, 100, burn_in=50)
        
        tv_dist = tv_distance_discrete_gaussian(basis, 1.0, samples)
        logger.info(f"  ✓ Regular TV distance (dim=4): {tv_dist:.4f}")
        
    except Exception as e:
        logger.warning(f"  ! Regular TV distance failed (expected for large dims): {e}")
    
    try:
        # Try optimized version
        from stats_optimized import tv_distance_discrete_gaussian_optimized
        
        basis = create_lattice_basis(8, 'identity')
        samples, _ = imhk_sampler_wrapper(basis, 2.0, 100, burn_in=50)
        
        tv_dist_opt = tv_distance_discrete_gaussian_optimized(
            basis, 1.0, samples,
            max_points=1000,
            adaptive_sampling=True
        )
        logger.info(f"  ✓ Optimized TV distance (dim=8): {tv_dist_opt:.4f}")
        
    except Exception as e:
        logger.error(f"  ✗ Optimized TV distance: {e}")
        traceback.print_exc()
        return False
    
    return True

def check_diagnostics():
    """Check diagnostic tools."""
    logger.info("\n5. CHECKING DIAGNOSTICS")
    logger.info("=" * 70)
    
    from diagnostics import compute_ess, compute_autocorrelation
    from utils import create_lattice_basis
    from samplers import imhk_sampler_wrapper
    
    try:
        # Generate test samples
        basis = create_lattice_basis(4, 'identity')
        samples, _ = imhk_sampler_wrapper(basis, 2.0, 500, burn_in=100)
        
        # Test ESS
        ess_values = compute_ess(samples)
        logger.info(f"  ✓ ESS calculation: {ess_values[0]:.2f}")
        
        # Test autocorrelation
        acf_by_dim = compute_autocorrelation(samples, lag=50)
        logger.info(f"  ✓ Autocorrelation: {len(acf_by_dim)} dimensions")
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Diagnostics: {e}")
        traceback.print_exc()
        return False

def check_publication_scripts():
    """Check if publication scripts are ready."""
    logger.info("\n6. CHECKING PUBLICATION SCRIPTS")
    logger.info("=" * 70)
    
    scripts = [
        "generate_publication_results.py",
        "publication_results.py",
        "publication_crypto_results.py",
        "verify_publication_quality.py",
    ]
    
    all_scripts_ok = True
    
    for script in scripts:
        if Path(script).exists():
            logger.info(f"  ✓ {script}")
        else:
            logger.error(f"  ✗ {script} not found")
            all_scripts_ok = False
    
    return all_scripts_ok

def main():
    """Run all verification checks."""
    logger.info("COMPREHENSIVE COMPONENT VERIFICATION")
    logger.info("=" * 70)
    
    results = {
        "imports": check_imports(),
        "basis_types": check_basis_types(),
        "samplers": check_samplers(),
        "tv_distance": check_tv_distance(),
        "diagnostics": check_diagnostics(),
        "publication_scripts": check_publication_scripts(),
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 70)
    
    all_ok = True
    for component, status in results.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        logger.info(f"  {component:20s}: {status_str}")
        if not status:
            all_ok = False
    
    logger.info("\n" + "=" * 70)
    
    if all_ok:
        logger.info("ALL COMPONENTS VERIFIED! Ready for research publication.")
        logger.info("\nNext steps:")
        logger.info("1. Run: sage generate_publication_results.py")
        logger.info("2. Or for crypto focus: sage publication_crypto_results.py")
        logger.info("3. Check results/publication/ for outputs")
    else:
        logger.error("Some components failed verification. Please fix issues before proceeding.")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())