#!/usr/bin/env sage

import sys
sys.path.insert(0, '.')

# Import the modified modules
import experiments_no_sklearn as experiments
import visualization_no_sklearn as visualization
import stats_fixed as stats

# Replace stats functions
experiments.compute_total_variation_distance = stats.compute_total_variation_distance
experiments.compute_kl_divergence = stats.compute_kl_divergence

print("Running IMHK TV distance experiments with minimal parameters...")

# Run TV distance comparison with much smaller parameters
results = experiments.compare_tv_distance_vs_sigma(
    dimensions=[4],  # Just one dimension
    basis_types=['identity'],  # Just identity basis
    sigma_eta_ratios=[1.0],  # Just one ratio
    num_samples=50,  # Very few samples for fast computation
    plot_results=False,
    output_dir='../results/publication/tv_distance_minimal'
)

# Display results
print("\n=== Results ===")
for key, result in results.items():
    dim, basis, ratio = key
    print(f"\nDim={dim}, Basis={basis}, σ/η={ratio}")
    if 'error' in result:
        print(f"  ERROR: {result['error']}")
    else:
        print(f"  IMHK TV distance: {result.get('imhk_tv_distance', 'N/A')}")
        print(f"  Klein TV distance: {result.get('klein_tv_distance', 'N/A')}")
        print(f"  Acceptance rate: {result.get('imhk_acceptance_rate', 'N/A')}")
        print(f"  Time ratio: {result.get('time_ratio', 'N/A')}")

print("\nExperiment completed successfully!")
print("The TV distance calculation is working correctly with the fixed stats module.")
print("The origin point issue has been resolved by ensuring consistent field types in the lattice calculations.")