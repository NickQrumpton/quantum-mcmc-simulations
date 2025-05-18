#!/usr/bin/env python3
"""
Test script for the compare_tv_distance_vs_sigma function.

This script tests the implementation with small parameters to ensure
the function is working correctly.
"""

from experiments import compare_tv_distance_vs_sigma
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_small_comparison():
    """Test with small parameters to verify functionality."""
    print("Testing compare_tv_distance_vs_sigma with small parameters...")
    
    # Test with small dimensions and fewer ratios for quick execution
    results = compare_tv_distance_vs_sigma(
        dimensions=[2, 3],  # Small dimensions for quick testing
        basis_types=['identity', 'skewed'],  # Skip ill-conditioned and q-ary
        sigma_eta_ratios=[0.5, 1.0, 2.0],  # Fewer ratios
        num_samples=100,  # Fewer samples for speed
        plot_results=True,
        output_dir='test_results/tv_comparison'
    )
    
    # Check that we got results
    print(f"\nGenerated {len(results)} result entries")
    
    # Print a sample result
    for key, value in results.items():
        print(f"\nConfiguration: {key}")
        print(f"  TV Ratio: {value.get('tv_ratio', 'N/A')}")
        print(f"  IMHK TV: {value.get('imhk_tv_distance', 'N/A')}")
        print(f"  Klein TV: {value.get('klein_tv_distance', 'N/A')}")
        break  # Just show one example
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_small_comparison()