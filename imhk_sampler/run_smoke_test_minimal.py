"""
Minimal smoke test to verify experiment pipeline works.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_minimal_smoke_test():
    """Run a minimal smoke test."""
    try:
        # Import all required modules
        logger.info("Testing imports...")
        from utils import create_lattice_basis
        from samplers import imhk_sampler
        from parameter_config import compute_smoothing_parameter
        from stats import compute_total_variation_distance
        from diagnostics import compute_ess
        logger.info("✓ All imports successful")
        
        # Test basic experiment
        logger.info("\nRunning minimal experiment...")
        dim = 2
        basis = create_lattice_basis(dim, "identity")
        eta = compute_smoothing_parameter(basis)
        sigma = 2.0 * eta
        num_samples = 100
        
        # Run sampler
        samples, metadata = imhk_sampler(B=basis, sigma=sigma, num_samples=num_samples)
        logger.info(f"✓ IMHK sampler completed: acceptance rate = {metadata['acceptance_rate']:.3f}")
        
        # Compute metrics
        tv_dist = compute_total_variation_distance(samples, sigma, basis)
        ess = compute_ess(samples)
        logger.info(f"✓ Metrics computed: TV distance = {tv_dist:.6f}, mean ESS = {ess[0]:.1f}")
        
        # Test with experiments.report
        logger.info("\nTesting experiment runner...")
        from experiments.report import ExperimentRunner
        
        runner = ExperimentRunner(output_dir="smoke_test_minimal")
        config = {
            'dimension': 2,
            'basis_type': 'identity',
            'ratio': 1.0
        }
        
        result = runner.run_single_experiment(config)
        logger.info(f"✓ Single experiment completed: TV = {result['tv_mean']:.6f}")
        
        logger.info("\n✓ All smoke tests passed!")
        logger.info("\nReady to run full experiments with:")
        logger.info("sage -python run_smoke_test.py --dimensions 2 --basis-types identity skewed ill-conditioned --ratios 0.5 1.0 2.0 --num-chains 1 --samples-per-chain 100 --output-dir smoke_test_results")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_minimal_smoke_test()
    sys.exit(0 if success else 1)