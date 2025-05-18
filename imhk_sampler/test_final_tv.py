#!/usr/bin/env sage

import sys
sys.path.insert(0, '.')

# Import the modified experiments module without sklearn
import experiments_no_sklearn as experiments
import stats_fixed as stats

# Replace the stats functions in experiments
experiments.compute_total_variation_distance = stats.compute_total_variation_distance
experiments.compute_kl_divergence = stats.compute_kl_divergence

# Run comprehensive test
results = experiments.compare_tv_distance_vs_sigma(
    dimensions=[4, 8],
    basis_types=['identity', 'skewed'],
    sigma_eta_ratios=[0.5, 1.0, 2.0],
    num_samples=500,
    plot_results=False,  # Disable plotting to avoid matplotlib issues
    output_dir='../results/publication/tv_distance_final'
)

# Generate summary
print("\n=== TV Distance Comparison Results ===")
print(f"Total experiments: {len(results)}")

# Calculate statistics by basis type
by_basis = {}
for key, result in results.items():
    if 'error' not in result:
        basis = result['basis_type']
        if basis not in by_basis:
            by_basis[basis] = {
                'tv_distances': [],
                'acceptance_rates': [],
                'time_ratios': []
            }
        
        if result.get('imhk_tv_distance') is not None:
            by_basis[basis]['tv_distances'].append(result['imhk_tv_distance'])
        if result.get('imhk_acceptance_rate') is not None:
            by_basis[basis]['acceptance_rates'].append(result['imhk_acceptance_rate'])
        if result.get('time_ratio') is not None:
            by_basis[basis]['time_ratios'].append(result['time_ratio'])

# Print summary statistics
for basis, metrics in by_basis.items():
    print(f"\n{basis.upper()} BASIS:")
    
    if metrics['tv_distances']:
        avg_tv = sum(metrics['tv_distances']) / len(metrics['tv_distances'])
        print(f"  Average TV distance: {avg_tv:.4f}")
    
    if metrics['acceptance_rates']:
        avg_acc = sum(metrics['acceptance_rates']) / len(metrics['acceptance_rates'])
        print(f"  Average acceptance rate: {avg_acc:.2%}")
    
    if metrics['time_ratios']:
        avg_ratio = sum(metrics['time_ratios']) / len(metrics['time_ratios'])
        print(f"  Average speedup vs Klein: {avg_ratio:.2f}x")

print("\nDetailed results saved to: ../results/publication/tv_distance_final/")