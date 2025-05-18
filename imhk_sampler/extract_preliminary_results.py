#!/usr/bin/env sage -python
"""
Extract preliminary results from any completed experiments.
This script is designed to work even with partial results.
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Extracting Preliminary Results")
print("="*50)

# Look for result files
result_files = [
    'crypto_test_results.json',
    'crypto_test_report.json',
    'optimized_smoke_test_results.json',
    'smoke_test_crypto_results/experiment_report.json',
    'crypto_publication_results/summary.json',
    'crypto_publication_results/all_results.csv'
]

found_results = []

for file_path in result_files:
    if os.path.exists(file_path):
        print(f"Found: {file_path}")
        found_results.append(file_path)

if not found_results:
    print("\nNo result files found. Please run one of the test scripts first.")
    exit(1)

# Extract data from JSON files
all_data = []

for file_path in found_results:
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract results based on file structure
                if isinstance(data, list):
                    # Direct list of results
                    all_data.extend(data)
                elif 'results' in data:
                    # Results nested in dictionary
                    all_data.extend(data['results'])
                elif 'summary' in data:
                    # Summary format
                    print(f"\nSummary from {file_path}:")
                    for key, value in data['summary'].items():
                        print(f"  {key}: {value}")
                        
        elif file_path.endswith('.csv'):
            # Load CSV data
            df = pd.read_csv(file_path)
            print(f"\nCSV data from {file_path}:")
            print(df.head())
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Process extracted data
if all_data:
    print(f"\nExtracted {len(all_data)} experiment results")
    
    # Create summary DataFrame
    summary_data = []
    for result in all_data:
        if isinstance(result, dict):
            summary_data.append({
                'basis_type': result.get('basis_type', 'unknown'),
                'dimension': result.get('dimension', 0),
                'sigma': result.get('sigma', 0),
                'acceptance_rate': result.get('acceptance_rate', 0),
                'status': result.get('status', 'unknown'),
                'time': result.get('elapsed_time', result.get('time', 0))
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        print("\nSummary Table:")
        print(df.to_string())
        
        # Save summary
        df.to_csv('preliminary_results_summary.csv', index=False)
        print("\nSaved summary to preliminary_results_summary.csv")
        
        # Create simple visualization
        try:
            plt.figure(figsize=(10, 6))
            
            # Plot acceptance rates by basis type
            basis_types = df['basis_type'].unique()
            for basis in basis_types:
                data = df[df['basis_type'] == basis]
                plt.scatter(data['dimension'], data['acceptance_rate'], label=basis, s=100)
            
            plt.xlabel('Dimension')
            plt.ylabel('Acceptance Rate')
            plt.title('IMHK Acceptance Rates by Basis Type')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('preliminary_acceptance_rates.png', dpi=150, bbox_inches='tight')
            print("Created plot: preliminary_acceptance_rates.png")
            
        except Exception as e:
            print(f"Could not create plot: {e}")

# Extract successful configurations
print("\n" + "="*50)
print("SUCCESSFUL CONFIGURATIONS")
print("="*50)

for result in all_data:
    if isinstance(result, dict) and result.get('status') == 'success':
        print(f"\n{result.get('basis_type', 'unknown')} (dim={result.get('dimension', 0)}):")
        print(f"  Sigma: {result.get('sigma', 0):.4f}")
        print(f"  Acceptance rate: {result.get('acceptance_rate', 0):.4f}")
        print(f"  Time: {result.get('elapsed_time', result.get('time', 0)):.2f}s")
        if 'tv_distance' in result and result['tv_distance'] is not None:
            print(f"  TV distance: {result['tv_distance']:.6f}")

# Recommendations
print("\n" + "="*50)
print("RECOMMENDATIONS")
print("="*50)

if all_data:
    successful = sum(1 for r in all_data if r.get('status') == 'success')
    total = len(all_data)
    success_rate = successful / total if total > 0 else 0
    
    print(f"Success rate: {successful}/{total} ({success_rate:.1%})")
    
    if success_rate > 0.8:
        print("\nSystem is working well. Ready for full experiments.")
    elif success_rate > 0.5:
        print("\nPartial success. Check failed configurations and adjust parameters.")
    else:
        print("\nMany failures detected. Run debug_crypto_test.py for details.")
else:
    print("No experiment data found. Run test scripts first.")

print("\nPreliminary extraction complete.")
print("Check generated files:")
print("- preliminary_results_summary.csv")
print("- preliminary_acceptance_rates.png")