#!/usr/bin/env sage

import sys
sys.path.insert(0, '.')

# Import the modified modules
import experiments_no_sklearn as experiments
import visualization_no_sklearn as visualization
import stats_fixed as stats

# Inject visualization functions into experiments module
experiments.plot_2d_samples = visualization.plot_2d_samples
experiments.plot_3d_samples = visualization.plot_3d_samples
experiments.plot_2d_projections = visualization.plot_2d_projections

# Replace stats functions
experiments.compute_total_variation_distance = stats.compute_total_variation_distance
experiments.compute_kl_divergence = stats.compute_kl_divergence

# Inject into globals so dynamic import can find them
experiments.__builtins__['plot_2d_samples'] = visualization.plot_2d_samples
experiments.__builtins__['plot_3d_samples'] = visualization.plot_3d_samples
experiments.__builtins__['plot_2d_projections'] = visualization.plot_2d_projections

print("Starting IMHK TV distance experiments...")

# Run TV distance comparison
results = experiments.compare_tv_distance_vs_sigma(
    dimensions=[4, 8],
    basis_types=['identity', 'skewed'],
    sigma_eta_ratios=[0.5, 1.0, 2.0],
    num_samples=200,  # Using fewer samples for faster testing
    plot_results=False,
    output_dir='../results/publication/tv_distance_final'
)

# Generate summary
print("\n=== TV Distance Comparison Results ===")
print(f"Total experiments: {len(results)}")

# Successful experiments
successful = sum(1 for r in results.values() if 'error' not in r)
print(f"Successful experiments: {successful}")
print(f"Failed experiments: {len(results) - successful}")

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
    print(f"\n{basis.upper()} BASIS RESULTS:")
    
    if metrics['tv_distances']:
        tv_values = metrics['tv_distances']
        print(f"  TV distances: {len(tv_values)} measurements")
        print(f"    Mean: {sum(tv_values)/len(tv_values):.4f}")
        print(f"    Min:  {min(tv_values):.4f}")
        print(f"    Max:  {max(tv_values):.4f}")
    
    if metrics['acceptance_rates']:
        acc_values = metrics['acceptance_rates']
        print(f"  Acceptance rates: {len(acc_values)} measurements")
        print(f"    Mean: {sum(acc_values)/len(acc_values):.2%}")
        print(f"    Min:  {min(acc_values):.2%}")
        print(f"    Max:  {max(acc_values):.2%}")
    
    if metrics['time_ratios']:
        time_values = metrics['time_ratios']
        print(f"  Time ratios (Klein/IMHK): {len(time_values)} measurements")
        print(f"    Mean: {sum(time_values)/len(time_values):.2f}x")
        print(f"    Min:  {min(time_values):.2f}x")
        print(f"    Max:  {max(time_values):.2f}x")

print("\n=== Per-configuration Results ===")
for key in sorted(results.keys()):
    result = results[key]
    dim, basis, ratio = key
    print(f"\nDim={dim}, Basis={basis}, σ/η={ratio}")
    if 'error' in result:
        print(f"  ERROR: {result['error']}")
    else:
        print(f"  IMHK TV distance: {result.get('imhk_tv_distance', 'N/A')}")
        print(f"  Klein TV distance: {result.get('klein_tv_distance', 'N/A')}")
        print(f"  Acceptance rate: {result.get('imhk_acceptance_rate', 'N/A')}")
        print(f"  Time ratio: {result.get('time_ratio', 'N/A')}")

print(f"\nResults saved to: ../results/publication/tv_distance_final/")