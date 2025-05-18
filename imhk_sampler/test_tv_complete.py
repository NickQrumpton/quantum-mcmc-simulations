#!/usr/bin/env sage

import sys
sys.path.insert(0, '.')

from experiments import compare_tv_distance_vs_sigma
import stats_fixed as stats
from utils import discrete_gaussian_pdf

# Monkey patch the fixed stats module
import experiments
experiments.compute_total_variation_distance = stats.compute_total_variation_distance

# Create output directory
import os
output_dir = '../results/publication/tv_distance_comparison'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

# Run comprehensive test
results = compare_tv_distance_vs_sigma(
    dimensions=[4, 8, 16],
    basis_types=['identity', 'skewed'],
    sigma_eta_ratios=[0.5, 1.0, 2.0],
    num_samples=500,
    plot_results=False,  # Disable plots to avoid sklearn dependency
    output_dir=output_dir
)

# Generate summary statistics
print("\n=== TV Distance Comparison Results ===")
print(f"Total experiments: {len(results)}")

# Calculate average TV distances by basis type
by_basis = {}
for key, result in results.items():
    if 'error' not in result:
        basis = result['basis_type']
        if basis not in by_basis:
            by_basis[basis] = []
        
        if 'imhk_tv_distance' in result:
            by_basis[basis].append(result['imhk_tv_distance'])

for basis, tv_distances in by_basis.items():
    if tv_distances:
        avg_tv = sum(tv_distances) / len(tv_distances)
        print(f"\nAverage TV distance for {basis} basis: {avg_tv:.4f}")

print("\nResults saved to:", output_dir)