"""
Run smoke test for cryptographic lattice bases.
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.report import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(description='Run IMHK smoke test with cryptographic lattices')
    parser.add_argument('--dimensions', type=int, nargs='+', default=[32, 64])  # Smaller dimensions for testing
    parser.add_argument('--basis-types', nargs='+', default=['q-ary', 'NTRU', 'PrimeCyclotomic'])
    parser.add_argument('--ratios', type=float, nargs='+', default=[1.5, 2.0])  # Fewer ratios
    parser.add_argument('--num-chains', type=int, default=1)  # Single chain for speed
    parser.add_argument('--samples-per-chain', type=int, default=100)  # Fewer samples for speed
    parser.add_argument('--output-dir', type=str, default='smoke_test_crypto_results')
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    # Configure with command line arguments
    runner.num_chains = args.num_chains
    runner.num_samples = args.samples_per_chain
    runner.ratios = args.ratios
    runner.dimensions = args.dimensions
    runner.basis_types = args.basis_types
    
    # Run experiments
    runner.run_complete_analysis()

if __name__ == "__main__":
    main()