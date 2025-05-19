#!/usr/bin/env sage -python
"""
Fixed publication results generator v3 FINAL with all type issues resolved.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import logging
from math import sqrt
from typing import Dict, List, Tuple, Optional, Any
import traceback

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import necessary modules
from json_serialization_utils import (
    NumpyJSONEncoder, 
    sanitize_data_for_json, 
    save_json_safely, 
    validate_json_serializable
)
from fixed_acceptance_rates_figure import create_acceptance_rates_figure_fixed

# Configure enhanced logging
def setup_logging(log_dir: Path):
    """Setup comprehensive logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_dir / 'fixed_publication_results_v3_final.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler with simple formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = None  # Will be initialized when class is instantiated


class FixedPublicationResultsGeneratorV3Final:
    """Fixed generator v3 FINAL that ensures all figures and data are complete."""
    
    def __init__(self, output_dir: str = "fixed_publication_results"):
        """Initialize the generator with enhanced configuration."""
        global logger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup comprehensive logging
        self.log_dir = self.output_dir / "logs"
        logger = setup_logging(self.log_dir)
        
        logger.info(f"Initialized generator v3 FINAL with output directory: {output_dir}")
        
        # Create all necessary subdirectories
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.data_dir = self.output_dir / "data"
        
        for dir_path in [self.figures_dir, self.tables_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Track metrics for comprehensive reporting
        self.metrics_tracker = {
            'experiments_run': 0,
            'experiments_successful': 0,
            'experiments_failed': 0,
            'figures_generated': [],
            'tables_generated': [],
            'files_generated': [],
            'errors': []
        }
    
    def calculate_metric_with_fallback(self, metric_func, *args, default=0.0, 
                                     metric_name="metric", **kwargs):
        """Calculate a metric with comprehensive error handling."""
        try:
            result = metric_func(*args, **kwargs)
            
            # Handle various return types
            if result is None:
                logger.warning(f"{metric_name} returned None, using default {default}")
                return default
            
            # Convert to scalar if needed
            if isinstance(result, (list, tuple, np.ndarray)):
                if len(result) == 1:
                    result = result[0]
                elif len(result) == 0:
                    return default
                else:
                    # Take first element for multi-element results
                    result = result[0]
                    logger.warning(f"{metric_name} returned array of length {len(result)}, using first element")
            
            # Ensure numeric type
            try:
                return float(result)
            except (ValueError, TypeError):
                logger.warning(f"{metric_name} could not be converted to float: {result}, using default")
                return default
                
        except Exception as e:
            logger.error(f"Error calculating {metric_name}: {e}")
            logger.debug(traceback.format_exc())
            self.metrics_tracker['errors'].append(f"{metric_name}: {str(e)}")
            return default
    
    def run_comprehensive_experiments(self):
        """Run all experiments with complete metric calculation and type safety."""
        logger.info("Starting comprehensive experiments v3 FINAL")
        
        # Import required modules
        try:
            from utils import create_lattice_basis
            from parameter_config import compute_smoothing_parameter
            from diagnostics import compute_ess
            from fixed_samplers_v2_final_patched import fixed_imhk_sampler_wrapper, fixed_klein_sampler_wrapper
            from fixed_tv_distance_calculation_v2 import compute_tv_distance_structured
        except ImportError as e:
            logger.error(f"Import error: {e}")
            logger.info("Attempting to use fallback modules...")
            from samplers import imhk_sampler_wrapper as fixed_imhk_sampler_wrapper
            from samplers import klein_sampler_wrapper as fixed_klein_sampler_wrapper
            from stats import compute_total_variation_distance as compute_tv_distance_structured
        
        # Define experiment configurations
        experiments = {
            'standard_lattices': [
                # (dimension, basis_type, sigma_ratios, num_samples)
                (4, 'identity', np.linspace(0.5, 3.0, 8), 5000),
                (8, 'identity', np.linspace(0.5, 3.0, 6), 3000),
                (16, 'identity', np.linspace(0.5, 2.5, 5), 2000),
                (32, 'identity', np.linspace(1.0, 2.0, 4), 1000),
            ],
            'cryptographic_lattices': [
                # Q-ary lattices
                (8, 'q-ary', np.linspace(0.8, 2.5, 5), 3000),
                (16, 'q-ary', np.linspace(0.8, 2.0, 4), 2000),
                (32, 'q-ary', np.linspace(1.0, 1.8, 3), 1000),
                # NTRU and PrimeCyclotomic
                (512, 'NTRU', np.linspace(0.9, 1.3, 3), 500),
                (683, 'PrimeCyclotomic', np.linspace(0.9, 1.3, 3), 500),
            ]
        }
        
        all_results = []
        
        for experiment_type, configs in experiments.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Running {experiment_type} experiments")
            logger.info('='*60)
            
            for dim, basis_type, sigma_ratios, num_samples in configs:
                logger.info(f"\nTesting {basis_type} (dimension={dim})")
                
                # Create basis with error handling
                try:
                    basis_info = create_lattice_basis(dim, basis_type)
                    
                    # Handle different basis types
                    if isinstance(basis_info, tuple):
                        poly_mod, q = basis_info
                        # Conservative sigma for structured lattices
                        base_sigma = float(sqrt(q) / 30)
                        sigmas = [base_sigma * ratio for ratio in sigma_ratios]
                        eta = base_sigma
                        logger.info(f"Structured lattice: q={q}, base_sigma={base_sigma:.4f}")
                    else:
                        eta = compute_smoothing_parameter(basis_info)
                        sigmas = [eta * ratio for ratio in sigma_ratios]
                        base_sigma = eta
                        logger.info(f"Standard lattice: eta={eta:.4f}")
                
                except Exception as e:
                    logger.error(f"Failed to create basis for {basis_type}: {e}")
                    continue
                
                for i, sigma in enumerate(sigmas):
                    sigma_ratio = sigma_ratios[i]
                    logger.info(f"\nExperiment: {basis_type} d={dim} σ/η={sigma_ratio:.2f}")
                    
                    self.metrics_tracker['experiments_run'] += 1
                    
                    # Initialize result dictionary with all fields
                    result = {
                        'experiment_type': experiment_type,
                        'basis_type': basis_type,
                        'dimension': dim,
                        'sigma': sigma,
                        'sigma_ratio': sigma_ratio,
                        'eta': eta,
                        'num_samples': num_samples,
                        'imhk_acceptance_rate': 0.0,
                        'imhk_time': 0.0,
                        'klein_time': 0.0,
                        'speedup': 0.0,
                        'ess': 0.0,
                        'ess_per_second': 0.0,
                        'tv_distance': None,
                        'tv_distance_klein': None,
                        'error': None
                    }
                    
                    try:
                        # Run IMHK sampler
                        start_time = time.time()
                        imhk_samples, imhk_metadata = fixed_imhk_sampler_wrapper(
                            basis_info=basis_info,
                            sigma=sigma,
                            num_samples=num_samples,
                            burn_in=min(1000, num_samples//2),
                            basis_type=basis_type
                        )
                        imhk_time = time.time() - start_time
                        
                        result['imhk_time'] = imhk_time
                        result['imhk_acceptance_rate'] = imhk_metadata.get('acceptance_rate', 0.0)
                        
                        logger.info(f"IMHK: {imhk_samples.shape[0]} samples, "
                                  f"acceptance={result['imhk_acceptance_rate']:.4f}, "
                                  f"time={imhk_time:.2f}s")
                        
                        # Run Klein sampler for comparison
                        klein_start = time.time()
                        klein_samples, klein_metadata = fixed_klein_sampler_wrapper(
                            basis_info=basis_info,
                            sigma=sigma,
                            num_samples=min(500, num_samples//2),
                            basis_type=basis_type
                        )
                        klein_time = time.time() - klein_start
                        
                        result['klein_time'] = klein_time
                        result['speedup'] = klein_time / imhk_time if imhk_time > 0 else 0.0
                        
                        logger.info(f"Klein: {klein_samples.shape[0]} samples, time={klein_time:.2f}s")
                        
                        # Compute ESS with robust error handling
                        if imhk_samples.shape[0] > 0:
                            ess = self.calculate_metric_with_fallback(
                                compute_ess, imhk_samples[:, 0], 
                                default=0.0, metric_name="ESS"
                            )
                            # Ensure ess is scalar
                            result['ess'] = max(0.0, float(ess))
                            result['ess_per_second'] = result['ess'] / imhk_time if imhk_time > 0 else 0.0
                        
                        # Compute TV distance
                        max_samples_for_tv = min(2000, imhk_samples.shape[0])
                        if max_samples_for_tv > 0:
                            tv_imhk = self.calculate_metric_with_fallback(
                                compute_tv_distance_structured,
                                imhk_samples[:max_samples_for_tv],
                                sigma, basis_info, basis_type,
                                default=None, metric_name="TV distance (IMHK)"
                            )
                            result['tv_distance'] = tv_imhk
                            
                            # Klein TV distance
                            max_klein_samples = min(500, klein_samples.shape[0])
                            if max_klein_samples > 0:
                                tv_klein = self.calculate_metric_with_fallback(
                                    compute_tv_distance_structured,
                                    klein_samples[:max_klein_samples],
                                    sigma, basis_info, basis_type,
                                    default=None, metric_name="TV distance (Klein)"
                                )
                                result['tv_distance_klein'] = tv_klein
                        
                        self.metrics_tracker['experiments_successful'] += 1
                        
                    except Exception as e:
                        logger.error(f"Experiment failed: {e}")
                        logger.debug(traceback.format_exc())
                        result['error'] = str(e)
                        self.metrics_tracker['experiments_failed'] += 1
                    
                    all_results.append(result)
                    
                    # Safe logging of result summary
                    tv_str = f"{result['tv_distance']:.4f}" if result['tv_distance'] is not None else "None"
                    logger.info(f"Result summary: acceptance={result['imhk_acceptance_rate']:.4f}, "
                              f"ESS={result['ess']:.2f}, TV={tv_str}")
        
        # Create DataFrame and save results
        logger.info(f"\nExperiment summary: {self.metrics_tracker['experiments_successful']} "
                   f"successful, {self.metrics_tracker['experiments_failed']} failed")
        
        df = pd.DataFrame(all_results)
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['sigma', 'sigma_ratio', 'eta', 'imhk_acceptance_rate', 
                          'imhk_time', 'klein_time', 'speedup', 'ess', 'ess_per_second']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Save results with proper error handling
        try:
            csv_path = self.data_dir / 'all_results.csv'
            df.to_csv(csv_path, index=False)
            self.metrics_tracker['files_generated'].append(str(csv_path))
            logger.info(f"Saved CSV results to {csv_path}")
            
            json_path = self.data_dir / 'all_results.json'
            save_json_safely(all_results, json_path)
            self.metrics_tracker['files_generated'].append(str(json_path))
            logger.info(f"Saved JSON results to {json_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        return df
    
    def create_publication_figures(self, df: pd.DataFrame):
        """Create all publication figures with enhanced error handling."""
        logger.info("Creating publication figures v3")
        
        # Set consistent style for all figures
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Figure 1: TV distance comparison
        try:
            logger.info("Creating TV distance figure")
            self._create_tv_distance_figure(df)
            logger.info("Successfully created fig1")
        except Exception as e:
            logger.error(f"Failed to create TV distance figure: {e}")
            self.metrics_tracker['errors'].append(f"Figure 1: {str(e)}")
        
        # Figure 2: Acceptance rates heatmap
        try:
            logger.info("Creating fixed acceptance rates heatmap")
            create_acceptance_rates_figure_fixed(df, self.figures_dir)
            logger.info("Successfully created fig2")
        except Exception as e:
            logger.error(f"Failed to create acceptance rates figure: {e}")
            self.metrics_tracker['errors'].append(f"Figure 2: {str(e)}")
        
        # Figure 3: Performance analysis
        try:
            logger.info("Creating performance figure")
            self._create_performance_figure(df)
            logger.info("Successfully created fig3")
        except Exception as e:
            logger.error(f"Failed to create performance figure: {e}")
            self.metrics_tracker['errors'].append(f"Figure 3: {str(e)}")
    
    def _create_tv_distance_figure(self, df: pd.DataFrame):
        """Create comprehensive TV distance comparison figure."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot for different dimensions
        dimensions = [4, 8, 16, 32]
        axes = [ax1, ax2, ax3, ax4]
        
        for dim, ax in zip(dimensions, axes):
            # Filter data for this dimension
            dim_data = df[df['dimension'] == dim]
            
            # Plot by basis type
            for basis_type in dim_data['basis_type'].unique():
                basis_data = dim_data[dim_data['basis_type'] == basis_type]
                
                # Filter out None values
                valid_data = basis_data[basis_data['tv_distance'].notna()]
                
                if len(valid_data) > 0:
                    ax.plot(valid_data['sigma_ratio'], 
                           valid_data['tv_distance'],
                           'o-', label=basis_type, markersize=6)
            
            ax.set_xlabel('σ/η ratio')
            ax.set_ylabel('TV Distance')
            ax.set_title(f'Dimension {dim}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        pdf_path = self.figures_dir / 'fig1_tv_distance_comparison.pdf'
        png_path = self.figures_dir / 'fig1_tv_distance_comparison.png'
        
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.metrics_tracker['figures_generated'].extend([str(pdf_path), str(png_path)])
        logger.info("Completed TV distance figure")
    
    def _create_performance_figure(self, df: pd.DataFrame):
        """Create comprehensive performance analysis figure."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # ESS vs dimension
        dim_group = df.groupby(['dimension', 'basis_type'])['ess'].mean().reset_index()
        for basis_type in dim_group['basis_type'].unique():
            basis_data = dim_group[dim_group['basis_type'] == basis_type]
            ax1.plot(basis_data['dimension'], basis_data['ess'], 
                    'o-', label=basis_type, markersize=6)
        
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Effective Sample Size')
        ax1.set_title('ESS vs Dimension')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Speedup analysis
        speedup_group = df.groupby(['dimension', 'basis_type'])['speedup'].mean().reset_index()
        for basis_type in speedup_group['basis_type'].unique():
            basis_data = speedup_group[speedup_group['basis_type'] == basis_type]
            ax2.plot(basis_data['dimension'], basis_data['speedup'], 
                    'o-', label=basis_type, markersize=6)
        
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Speedup (Klein/IMHK)')
        ax2.set_title('Relative Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ESS per second
        ess_rate_group = df.groupby(['dimension', 'basis_type'])['ess_per_second'].mean().reset_index()
        for basis_type in ess_rate_group['basis_type'].unique():
            basis_data = ess_rate_group[ess_rate_group['basis_type'] == basis_type]
            ax3.plot(basis_data['dimension'], basis_data['ess_per_second'], 
                    'o-', label=basis_type, markersize=6)
        
        ax3.set_xlabel('Dimension')
        ax3.set_ylabel('ESS per Second')
        ax3.set_title('Sampling Efficiency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Acceptance rate vs sigma ratio
        for basis_type in df['basis_type'].unique():
            basis_data = df[df['basis_type'] == basis_type]
            avg_acceptance = basis_data.groupby('sigma_ratio')['imhk_acceptance_rate'].mean()
            ax4.plot(avg_acceptance.index, avg_acceptance.values, 
                    'o-', label=basis_type, markersize=6)
        
        ax4.set_xlabel('σ/η ratio')
        ax4.set_ylabel('Acceptance Rate')
        ax4.set_title('Acceptance Rate vs σ/η')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        pdf_path = self.figures_dir / 'fig3_performance_analysis.pdf'
        png_path = self.figures_dir / 'fig3_performance_analysis.png'
        
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.metrics_tracker['figures_generated'].extend([str(pdf_path), str(png_path)])
        logger.info("Completed performance figure")
    
    def create_publication_tables(self, df: pd.DataFrame):
        """Create publication-ready tables with enhanced formatting."""
        logger.info("Creating publication tables")
        
        # Table 1: Summary statistics by lattice type
        try:
            summary_stats = df.groupby('basis_type').agg({
                'imhk_acceptance_rate': ['mean', 'std'],
                'ess': ['mean', 'std'],
                'tv_distance': ['count', 'mean', 'std'],
                'speedup': ['mean', 'std']
            }).round(4)
            
            summary_path_csv = self.tables_dir / 'table1_summary_statistics.csv'
            summary_path_tex = self.tables_dir / 'table1_summary_statistics.tex'
            
            summary_stats.to_csv(summary_path_csv)
            summary_stats.to_latex(summary_path_tex, escape=False)
            
            self.metrics_tracker['tables_generated'].extend([
                str(summary_path_csv), str(summary_path_tex)
            ])
            logger.info(f"Created {summary_path_csv}")
            
        except Exception as e:
            logger.error(f"Failed to create Table 1: {e}")
            self.metrics_tracker['errors'].append(f"Table 1: {str(e)}")
        
        # Table 2: Optimal parameters by lattice type and dimension
        try:
            optimal_params = []
            
            for basis_type in df['basis_type'].unique():
                for dim in df['dimension'].unique():
                    subset = df[(df['basis_type'] == basis_type) & 
                               (df['dimension'] == dim)]
                    
                    if len(subset) > 0:
                        # Find best parameters based on ESS
                        best_idx = subset['ess'].idxmax()
                        best_row = subset.loc[best_idx]
                        
                        optimal_params.append({
                            'basis_type': basis_type,
                            'dimension': dim,
                            'optimal_sigma_ratio': best_row['sigma_ratio'],
                            'acceptance_rate': best_row['imhk_acceptance_rate'],
                            'ess': best_row['ess'],
                            'tv_distance': best_row['tv_distance']
                        })
            
            optimal_df = pd.DataFrame(optimal_params)
            
            optimal_path_csv = self.tables_dir / 'table2_optimal_parameters.csv'
            optimal_path_tex = self.tables_dir / 'table2_optimal_parameters.tex'
            
            optimal_df.to_csv(optimal_path_csv, index=False)
            optimal_df.to_latex(optimal_path_tex, index=False, escape=False)
            
            self.metrics_tracker['tables_generated'].extend([
                str(optimal_path_csv), str(optimal_path_tex)
            ])
            logger.info(f"Created {optimal_path_csv}")
            
        except Exception as e:
            logger.error(f"Failed to create Table 2: {e}")
            self.metrics_tracker['errors'].append(f"Table 2: {str(e)}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive report of all results and metrics."""
        logger.info("Generating publication report")
        
        report = {
            'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics_summary': self.metrics_tracker,
            'output_files': {
                'figures': self.metrics_tracker['figures_generated'],
                'tables': self.metrics_tracker['tables_generated'],
                'data': self.metrics_tracker['files_generated']
            },
            'experiment_summary': {
                'total_experiments': self.metrics_tracker['experiments_run'],
                'successful': self.metrics_tracker['experiments_successful'],
                'failed': self.metrics_tracker['experiments_failed'],
                'success_rate': (self.metrics_tracker['experiments_successful'] / 
                               self.metrics_tracker['experiments_run'] * 100 
                               if self.metrics_tracker['experiments_run'] > 0 else 0)
            },
            'errors_encountered': self.metrics_tracker['errors']
        }
        
        # Save report
        report_path = self.output_dir / 'publication_report.json'
        save_json_safely(report, report_path)
        
        # Create summary text file
        summary_path = self.output_dir / 'summary_report.txt'
        with open(summary_path, 'w') as f:
            f.write("Publication Results Generation Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated on: {report['generation_timestamp']}\n\n")
            
            f.write("Experiments Summary:\n")
            f.write(f"  Total: {report['experiment_summary']['total_experiments']}\n")
            f.write(f"  Successful: {report['experiment_summary']['successful']}\n")
            f.write(f"  Failed: {report['experiment_summary']['failed']}\n")
            f.write(f"  Success Rate: {report['experiment_summary']['success_rate']:.1f}%\n\n")
            
            f.write("Generated Files:\n")
            f.write(f"  Figures: {len(report['output_files']['figures'])}\n")
            f.write(f"  Tables: {len(report['output_files']['tables'])}\n")
            f.write(f"  Data Files: {len(report['output_files']['data'])}\n\n")
            
            if report['errors_encountered']:
                f.write("Errors Encountered:\n")
                for error in report['errors_encountered']:
                    f.write(f"  - {error}\n")
            else:
                f.write("No errors encountered during generation.\n")
        
        logger.info(f"Report saved to {report_path}")
        logger.info(f"Summary saved to {summary_path}")
    
    def generate_results(self):
        """Main method to generate all publication results."""
        logger.info("Starting comprehensive publication results generation V3 FINAL")
        
        try:
            # Run experiments
            df = self.run_comprehensive_experiments()
            
            # Create figures
            self.create_publication_figures(df)
            
            # Create tables
            self.create_publication_tables(df)
            
            # Generate report
            self.generate_comprehensive_report()
            
            logger.info("Successfully completed publication results generation")
            
        except Exception as e:
            logger.error(f"Fatal error in result generation: {e}")
            logger.error(traceback.format_exc())
            self.metrics_tracker['errors'].append(f"Fatal: {str(e)}")
            self.generate_comprehensive_report()
            raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate publication-quality results')
    parser.add_argument('--output-dir', default='fixed_publication_results',
                      help='Output directory for results')
    parser.add_argument('--all', action='store_true',
                      help='Run all experiments (default: run subset)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up logging
    global logger
    logger = logging.getLogger(__name__)
    
    # Create generator and run
    generator = FixedPublicationResultsGeneratorV3Final(args.output_dir)
    
    try:
        generator.generate_results()
        logger.info("Successfully generated results")
    except Exception as e:
        logger.error(f"Failed to generate results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()