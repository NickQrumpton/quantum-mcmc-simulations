#!/usr/bin/env sage

import sys
sys.path.insert(0, '.')

# Import the fixed modules
import experiments_fixed as experiments
import visualization_no_sklearn as visualization
import stats_fixed as stats
import diagnostics_fixed as diagnostics

# Replace modules in experiments
experiments.compute_total_variation_distance = stats.compute_total_variation_distance
experiments.compute_kl_divergence = stats.compute_kl_divergence
experiments.compute_autocorrelation = diagnostics.compute_autocorrelation
experiments.compute_ess = diagnostics.compute_ess
experiments.plot_trace = diagnostics.plot_trace
experiments.plot_autocorrelation = diagnostics.plot_autocorrelation
experiments.plot_acceptance_trace = diagnostics.plot_acceptance_trace

print("Running optimized IMHK TV distance experiments...")

# Run with minimal parameters for fast execution
results = experiments.compare_tv_distance_vs_sigma(
    dimensions=[4, 8],  # Just two dimensions
    basis_types=['identity', 'skewed'],  # Two basis types
    sigma_eta_ratios=[0.5, 1.0, 2.0],  # Three ratios
    num_samples=200,  # Small sample size
    plot_results=False,  # Disable plotting
    output_dir='../results/publication/tv_distance_optimized'
)

# Generate summary
print("\n=== TV Distance Comparison Results ===")
print(f"Total experiments: {len(results)}")

# Count successful experiments
successful = sum(1 for r in results.values() if 'error' not in r)
failed = len(results) - successful

print(f"Successful experiments: {successful}")
print(f"Failed experiments: {failed}")

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

# Display a few detailed results
print("\n=== Sample Detailed Results ===")
for i, (key, result) in enumerate(sorted(results.items())[:3]):
    dim, basis, ratio = key
    print(f"\nDim={dim}, Basis={basis}, σ/η={ratio}")
    if 'error' not in result:
        print(f"  IMHK TV distance: {result.get('imhk_tv_distance', 'N/A'):.4f}")
        print(f"  Klein TV distance: {result.get('klein_tv_distance', 'N/A'):.4f}")
        print(f"  Acceptance rate: {result.get('imhk_acceptance_rate', 'N/A'):.2%}")
        print(f"  Time ratio: {result.get('time_ratio', 'N/A'):.2f}x")
    else:
        print(f"  ERROR: {result['error']}")

print(f"\nFull results saved to: ../results/publication/tv_distance_optimized/")
print("Experiment completed successfully within time limits!")