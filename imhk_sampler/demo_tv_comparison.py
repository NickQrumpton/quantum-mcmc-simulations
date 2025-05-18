#!/usr/bin/env python3
"""
Demo script showing how to use the compare_tv_distance_vs_sigma function.

This demonstrates typical usage patterns for the function.
"""

from experiments import compare_tv_distance_vs_sigma
import json

def main():
    """Run a demonstration of TV distance comparison."""
    
    # Example 1: Comparing different basis types across dimensions
    print("Example 1: Comparing basis types across dimensions")
    print("-" * 50)
    
    results = compare_tv_distance_vs_sigma(
        dimensions=[4, 8],  # Small dimensions for demo
        basis_types=['identity', 'skewed', 'ill-conditioned'],
        sigma_eta_ratios=[0.5, 1.0, 2.0, 4.0],  # Key ratios
        num_samples=500,  # Moderate sample size
        plot_results=True,
        output_dir='demo_results/basis_comparison'
    )
    
    print(f"Generated {len(results)} configurations")
    
    # Analyze results for dimension 4
    print("\nResults for dimension 4:")
    for basis_type in ['identity', 'skewed', 'ill-conditioned']:
        tv_ratios = []
        for ratio in [0.5, 1.0, 2.0, 4.0]:
            key = (4, basis_type, ratio)
            if key in results and 'tv_ratio' in results[key]:
                tv_ratios.append(results[key]['tv_ratio'])
        if tv_ratios:
            avg_ratio = sum(tv_ratios) / len(tv_ratios)
            print(f"  {basis_type}: Average TV ratio = {avg_ratio:.4f}")
    
    # Example 2: High-dimensional comparison (for research)
    print("\n\nExample 2: High-dimensional lattices")
    print("-" * 50)
    
    # For actual research, you might use:
    # results_hd = compare_tv_distance_vs_sigma(
    #     dimensions=[16, 32, 64],
    #     basis_types=['identity', 'ill-conditioned'],
    #     sigma_eta_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
    #     num_samples=2000,
    #     plot_results=True,
    #     output_dir='research_results/high_dim'
    # )
    
    print("(High-dimensional example commented out for demo)")
    
    # Example 3: Custom output analysis
    print("\n\nExample 3: Custom analysis of results")
    print("-" * 50)
    
    # Find configurations where IMHK significantly outperforms Klein
    significant_improvements = []
    for key, result in results.items():
        if 'tv_ratio' in result and result['tv_ratio'] is not None:
            if result['tv_ratio'] < 0.8:  # 20% improvement
                significant_improvements.append({
                    'config': key,
                    'ratio': result['tv_ratio'],
                    'imhk_tv': result['imhk_tv_distance'],
                    'klein_tv': result['klein_tv_distance']
                })
    
    if significant_improvements:
        print(f"Found {len(significant_improvements)} configurations with >20% improvement:")
        for item in significant_improvements[:3]:  # Show first 3
            dim, basis, sigma_ratio = item['config']
            print(f"  Dim={dim}, Basis={basis}, σ/η={sigma_ratio}: "
                  f"TV ratio={item['ratio']:.4f}")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()