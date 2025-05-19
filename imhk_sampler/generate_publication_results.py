#!/usr/bin/env sage -python
"""
Generate comprehensive publication-quality results for IMHK sampler research paper.
This script produces all figures, tables, and data needed for publication.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import logging
from math import sqrt

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import JSON serialization utilities
from json_serialization_utils import (
    NumpyJSONEncoder, 
    sanitize_data_for_json, 
    save_json_safely, 
    validate_json_serializable
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.figsize': (8, 6),
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf'
})

class PublicationResultsGenerator:
    """Generate publication-quality results for IMHK sampler."""
    
    def __init__(self, output_dir="publication_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(exist_ok=True)
    
    def run_comprehensive_experiments(self):
        """Run all experiments needed for publication."""
        from utils import create_lattice_basis
        from samplers import imhk_sampler_wrapper, klein_sampler_wrapper
        from parameter_config import compute_smoothing_parameter
        from stats import compute_total_variation_distance
        from diagnostics import compute_ess
        
        # Define experiment configurations
        experiments = {
            'standard_lattices': [
                # (dimension, basis_type, sigma_ratios, num_samples)
                (4, 'identity', np.linspace(0.5, 3.0, 10), 5000),
                (8, 'identity', np.linspace(0.5, 3.0, 8), 3000),
                (16, 'identity', np.linspace(0.5, 2.5, 6), 2000),
                (32, 'identity', np.linspace(1.0, 2.0, 4), 1000),
            ],
            'cryptographic_lattices': [
                (8, 'q-ary', np.linspace(0.5, 3.0, 8), 3000),
                (16, 'q-ary', np.linspace(0.5, 2.5, 6), 2000),
                (32, 'q-ary', np.linspace(1.0, 2.0, 4), 1000),
                (512, 'NTRU', np.linspace(0.8, 1.5, 5), 1000),
                (683, 'PrimeCyclotomic', np.linspace(0.8, 1.5, 5), 1000),
            ]
        }
        
        all_results = []
        
        for experiment_type, configs in experiments.items():
            logger.info(f"\nRunning {experiment_type} experiments...")
            
            for dim, basis_type, sigma_ratios, num_samples in configs:
                # Create basis
                basis_info = create_lattice_basis(dim, basis_type)
                
                # Handle different basis types
                if isinstance(basis_info, tuple):
                    poly_mod, q = basis_info
                    base_sigma = float(sqrt(q) / 20)
                    sigmas = [base_sigma * ratio for ratio in sigma_ratios]
                    eta = None
                else:
                    eta = compute_smoothing_parameter(basis_info)
                    sigmas = [eta * ratio for ratio in sigma_ratios]
                    base_sigma = eta
                
                for i, sigma in enumerate(sigmas):
                    sigma_ratio = sigma_ratios[i]
                    logger.info(f"Testing {basis_type} (d={dim}) σ/η={sigma_ratio:.2f}")
                    
                    try:
                        # Run IMHK sampler
                        start_time = time.time()
                        imhk_samples, imhk_metadata = imhk_sampler_wrapper(
                            basis_info=basis_info,
                            sigma=sigma,
                            num_samples=num_samples,
                            burn_in=min(1000, num_samples//2),
                            basis_type=basis_type
                        )
                        imhk_time = time.time() - start_time
                        
                        # Run Klein sampler for comparison (fewer samples)
                        klein_start = time.time()
                        klein_samples, klein_metadata = klein_sampler_wrapper(
                            basis_info=basis_info,
                            sigma=sigma,
                            num_samples=min(500, num_samples//2),
                            basis_type=basis_type
                        )
                        klein_time = time.time() - klein_start
                        
                        result = {
                            'experiment_type': experiment_type,
                            'basis_type': basis_type,
                            'dimension': dim,
                            'sigma': sigma,
                            'sigma_ratio': sigma_ratio,
                            'eta': eta,
                            'num_samples': num_samples,
                            'imhk_acceptance_rate': imhk_metadata.get('acceptance_rate', 0),
                            'imhk_time': imhk_time,
                            'klein_time': klein_time,
                            'speedup': klein_time / imhk_time if imhk_time > 0 else None
                        }
                        
                        # Compute ESS
                        try:
                            ess = compute_ess(imhk_samples[:, 0])
                            result['ess'] = ess
                            result['ess_per_second'] = ess / imhk_time
                        except:
                            result['ess'] = None
                            result['ess_per_second'] = None
                        
                        # Compute TV distance for smaller dimensions
                        if not isinstance(basis_info, tuple) and dim <= 16:
                            try:
                                tv_imhk = compute_total_variation_distance(
                                    imhk_samples[:min(1000, len(imhk_samples))],
                                    sigma, basis_info,
                                    max_radius=max(2, int(3.0/np.sqrt(dim)))
                                )
                                result['tv_distance'] = tv_imhk
                                
                                # Also compute for Klein
                                tv_klein = compute_total_variation_distance(
                                    klein_samples[:min(500, len(klein_samples))],
                                    sigma, basis_info,
                                    max_radius=max(2, int(3.0/np.sqrt(dim)))
                                )
                                result['tv_distance_klein'] = tv_klein
                                
                            except:
                                result['tv_distance'] = None
                                result['tv_distance_klein'] = None
                        else:
                            result['tv_distance'] = None
                            result['tv_distance_klein'] = None
                        
                        all_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Experiment failed: {e}")
        
        # Save results
        df = pd.DataFrame(all_results)
        df.to_csv(self.data_dir / 'all_results.csv', index=False)
        
        # Save JSON safely with proper type conversion
        save_json_safely(all_results, self.data_dir / 'all_results.json')
        
        return df
    
    def create_figure_1_tv_distance_comparison(self, df):
        """Figure 1: TV distance vs sigma/eta ratio comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Panel A: Identity lattices
        identity_data = df[df['basis_type'] == 'identity']
        for dim in sorted(identity_data['dimension'].unique()):
            data = identity_data[identity_data['dimension'] == dim]
            valid_data = data[data['tv_distance'].notna()]
            if not valid_data.empty:
                ax1.plot(valid_data['sigma_ratio'], valid_data['tv_distance'],
                        marker='o', label=f'd = {dim}')
        
        ax1.set_xlabel('σ/η Ratio')
        ax1.set_ylabel('Total Variation Distance')
        ax1.set_title('(a) Identity Lattices')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 3)
        ax1.set_ylim(0, 1)
        
        # Panel B: Cryptographic lattices
        crypto_data = df[df['basis_type'].isin(['q-ary', 'NTRU', 'PrimeCyclotomic'])]
        for basis in crypto_data['basis_type'].unique():
            data = crypto_data[crypto_data['basis_type'] == basis]
            valid_data = data[data['tv_distance'].notna()]
            if not valid_data.empty:
                ax2.plot(valid_data['sigma_ratio'], valid_data['tv_distance'],
                        marker='s', label=basis)
        
        ax2.set_xlabel('σ/η Ratio')
        ax2.set_ylabel('Total Variation Distance')
        ax2.set_title('(b) Cryptographic Lattices')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig1_tv_distance_comparison.pdf')
        plt.close()
    
    def create_figure_2_acceptance_rates(self, df):
        """Figure 2: Acceptance rates across different lattice types."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data for heatmap
        pivot_data = df.pivot_table(
            values='imhk_acceptance_rate',
            index='basis_type',
            columns='sigma_ratio',
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Acceptance Rate'})
        
        ax.set_xlabel('σ/η Ratio')
        ax.set_ylabel('Lattice Type')
        ax.set_title('IMHK Acceptance Rates')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig2_acceptance_rates_heatmap.pdf')
        plt.close()
    
    def create_figure_3_performance_analysis(self, df):
        """Figure 3: Performance analysis (ESS, speedup)."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
        
        # Panel A: ESS by dimension
        ess_data = df[df['ess'].notna()]
        for basis in ess_data['basis_type'].unique():
            data = ess_data[ess_data['basis_type'] == basis]
            ax1.scatter(data['dimension'], data['ess'], s=80, label=basis)
        
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Effective Sample Size')
        ax1.set_title('(a) ESS vs Dimension')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Panel B: ESS per second
        for basis in ess_data['basis_type'].unique():
            data = ess_data[ess_data['basis_type'] == basis]
            ax2.scatter(data['dimension'], data['ess_per_second'], s=80, label=basis)
        
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('ESS per Second')
        ax2.set_title('(b) Sampling Efficiency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Panel C: Runtime comparison
        runtime_by_dim = df.groupby(['dimension', 'basis_type'])['imhk_time'].mean().reset_index()
        for basis in runtime_by_dim['basis_type'].unique():
            data = runtime_by_dim[runtime_by_dim['basis_type'] == basis]
            ax3.plot(data['dimension'], data['imhk_time'], marker='o', label=basis)
        
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('Runtime (seconds)')
        ax3.set_title('(c) IMHK Runtime Scaling')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Panel D: Speedup over Klein
        speedup_data = df[df['speedup'].notna()]
        for basis in speedup_data['basis_type'].unique():
            data = speedup_data[speedup_data['basis_type'] == basis]
            ax4.scatter(data['dimension'], data['speedup'], s=80, label=basis)
        
        ax4.set_xlabel('Dimension')
        ax4.set_ylabel('Speedup Factor')
        ax4.set_title('(d) IMHK Speedup over Klein')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'fig3_performance_analysis.pdf')
        plt.close()
    
    def create_table_1_summary_statistics(self, df):
        """Table 1: Summary statistics for all experiments."""
        summary = df.groupby(['basis_type', 'dimension']).agg({
            'imhk_acceptance_rate': ['mean', 'std'],
            'tv_distance': ['mean', 'std'],
            'ess': 'mean',
            'imhk_time': 'mean',
            'speedup': 'mean'
        }).round(4)
        
        # Convert to LaTeX
        latex_table = summary.to_latex(
            caption="Summary statistics for IMHK sampler across different lattice types",
            label="tab:summary_stats"
        )
        
        with open(self.tables_dir / 'table1_summary_statistics.tex', 'w') as f:
            f.write(latex_table)
        
        # Also save as CSV
        summary.to_csv(self.tables_dir / 'table1_summary_statistics.csv')
        
        return summary
    
    def create_table_2_optimal_parameters(self, df):
        """Table 2: Optimal sigma/eta ratios for each lattice type."""
        # Find optimal sigma ratio (maximizing acceptance rate while maintaining low TV distance)
        optimal_params = []
        
        for basis in df['basis_type'].unique():
            basis_data = df[df['basis_type'] == basis]
            
            # Find best trade-off between acceptance rate and TV distance
            if basis_data['tv_distance'].notna().any():
                # For lattices with TV distance
                valid_data = basis_data[basis_data['tv_distance'].notna()]
                
                # Compute score: high acceptance rate, low TV distance
                valid_data['score'] = valid_data['imhk_acceptance_rate'] / (valid_data['tv_distance'] + 0.01)
                
                best_idx = valid_data['score'].idxmax()
                best_row = valid_data.loc[best_idx]
            else:
                # For structured lattices, just maximize acceptance rate
                best_idx = basis_data['imhk_acceptance_rate'].idxmax()
                best_row = basis_data.loc[best_idx]
            
            optimal_params.append({
                'Lattice Type': basis,
                'Dimension': best_row['dimension'],
                'Optimal σ/η': best_row['sigma_ratio'],
                'Acceptance Rate': best_row['imhk_acceptance_rate'],
                'TV Distance': best_row.get('tv_distance', 'N/A'),
                'ESS': best_row.get('ess', 'N/A')
            })
        
        optimal_df = pd.DataFrame(optimal_params)
        
        # Convert to LaTeX
        latex_table = optimal_df.to_latex(
            index=False,
            caption="Optimal σ/η ratios for different lattice types",
            label="tab:optimal_params",
            column_format='l'*len(optimal_df.columns),
            float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x)
        )
        
        with open(self.tables_dir / 'table2_optimal_parameters.tex', 'w') as f:
            f.write(latex_table)
        
        # Also save as CSV
        optimal_df.to_csv(self.tables_dir / 'table2_optimal_parameters.csv', index=False)
        
        return optimal_df
    
    def generate_all_results(self):
        """Generate all publication results."""
        logger.info("Starting publication results generation...")
        
        # Run experiments
        logger.info("Running comprehensive experiments...")
        df = self.run_comprehensive_experiments()
        
        # Create figures
        logger.info("Creating publication figures...")
        self.create_figure_1_tv_distance_comparison(df)
        self.create_figure_2_acceptance_rates(df)
        self.create_figure_3_performance_analysis(df)
        
        # Create tables
        logger.info("Creating publication tables...")
        summary_stats = self.create_table_1_summary_statistics(df)
        optimal_params = self.create_table_2_optimal_parameters(df)
        
        # Generate final report
        logger.info("Generating final report...")
        
        # Convert data types to ensure JSON compatibility
        lattice_types = [str(lt) for lt in df['basis_type'].unique()]
        dimensions = [int(d) for d in sorted(df['dimension'].unique())]
        
        # Handle grouped data with proper type conversion
        acceptance_rates = {}
        for basis, rate in df.groupby('basis_type')['imhk_acceptance_rate'].mean().items():
            acceptance_rates[str(basis)] = float(rate) if pd.notna(rate) else None
            
        tv_distances = {}
        tv_df = df[df['tv_distance'].notna()]
        for basis, dist in tv_df.groupby('basis_type')['tv_distance'].mean().items():
            tv_distances[str(basis)] = float(dist) if pd.notna(dist) else None
        
        report = {
            'total_experiments': int(len(df)),
            'lattice_types': lattice_types,
            'dimensions_tested': dimensions,
            'average_acceptance_rates': acceptance_rates,
            'average_tv_distances': tv_distances,
            'optimal_parameters': optimal_params.to_dict('records')
        }
        
        # Validate and save JSON safely
        problems = validate_json_serializable(report)
        if problems:
            logger.warning(f"JSON serialization issues found: {problems}")
            
        save_json_safely(report, self.output_dir / 'publication_report.json')
        
        logger.info("\n" + "="*60)
        logger.info("PUBLICATION RESULTS GENERATED")
        logger.info("="*60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("\nFigures generated:")
        logger.info("- fig1_tv_distance_comparison.pdf")
        logger.info("- fig2_acceptance_rates_heatmap.pdf")
        logger.info("- fig3_performance_analysis.pdf")
        logger.info("\nTables generated:")
        logger.info("- table1_summary_statistics.tex/csv")
        logger.info("- table2_optimal_parameters.tex/csv")
        logger.info("\nData files:")
        logger.info("- all_results.csv")
        logger.info("- all_results.json")
        logger.info("- publication_report.json")
        
        return report

def main():
    """Generate all publication results."""
    generator = PublicationResultsGenerator()
    report = generator.generate_all_results()
    
    logger.info("\n🎉 Publication results successfully generated!")
    logger.info("Ready for inclusion in research paper.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())