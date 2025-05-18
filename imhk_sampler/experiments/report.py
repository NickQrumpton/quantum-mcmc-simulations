"""
Comprehensive experiment framework for IMHK sampler publication-quality results.

This module drives the main experiments for comparing TV distance across:
- Multiple basis types (identity, skewed, ill-conditioned)
- Various dimensions (2D, 4D, higher as needed)
- Different σ/η ratios
"""

import numpy as np
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure matplotlib for publication-quality plots
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 11
rcParams['figure.figsize'] = (8, 6)
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stats import compute_total_variation_distance
from parameter_config import compute_smoothing_parameter, generate_experiment_configs
from samplers import imhk_sampler, imhk_sampler_wrapper
from utils import create_lattice_basis
from diagnostics import compute_ess


class ExperimentRunner:
    """Manages comprehensive experiments for IMHK sampler evaluation."""
    
    def __init__(self, output_dir="results/experiments"):
        """
        Initialize experiment runner.
        
        Parameters
        ----------
        output_dir : str or Path
            Directory for saving results and plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.output_dir / "data"
        self.plot_dir = self.output_dir / "plots"
        self.data_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)
        
        # Configure experiment parameters
        self.num_chains = 5  # Number of independent chains
        self.num_samples = 10000  # Samples per chain
        self.ratios = [0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0]
        self.dimensions = [2, 4]  # Can extend to [2, 4, 8] for more
        self.basis_types = ["identity", "skewed", "ill_conditioned"]
        
        # Results storage
        self.results = []
        
    def run_single_experiment(self, config):
        """
        Run a single experiment configuration.
        
        Parameters
        ----------
        config : dict
            Experiment configuration with basis, sigma, dimension, etc.
        
        Returns
        -------
        dict
            Results including TV distance, ESS, and timing information
        """
        logger.info(f"Running experiment: dim={config['dimension']}, "
                   f"basis={config['basis_type']}, ratio={config['ratio']:.2f}")
        
        # Extract configuration
        dimension = config['dimension']
        basis_type = config['basis_type']
        ratio = config['ratio']
        
        # Create lattice basis
        lattice_basis = create_lattice_basis(dimension, basis_type)
        
        # Compute smoothing parameter and sigma
        eta = compute_smoothing_parameter(lattice_basis)
        sigma = ratio * eta
        
        # Run multiple chains
        tv_distances = []
        ess_values = []
        acceptance_rates = []
        
        for chain_idx in range(self.num_chains):
            logger.debug(f"Running chain {chain_idx + 1}/{self.num_chains}")
            
            # Generate samples
            try:
                samples, metadata = imhk_sampler_wrapper(
                    basis_info=lattice_basis,
                    sigma=sigma,
                    num_samples=self.num_samples,
                    basis_type=basis_type
                )
                
                # Compute TV distance
                tv_dist = compute_total_variation_distance(
                    samples, sigma, lattice_basis
                )
                
                if not np.isnan(tv_dist):
                    tv_distances.append(tv_dist)
                
                # Compute ESS
                ess = compute_ess(samples)
                ess_values.append(np.mean(ess))
                
                # Get acceptance rate from metadata
                if 'acceptance_rate' in metadata:
                    acceptance_rates.append(metadata['acceptance_rate'])
                    
            except Exception as e:
                logger.error(f"Error in chain {chain_idx}: {e}")
                continue
        
        # Aggregate results
        result = {
            'dimension': dimension,
            'basis_type': basis_type,
            'ratio': ratio,
            'sigma': sigma,
            'eta': eta,
            'tv_mean': np.mean(tv_distances) if tv_distances else np.nan,
            'tv_std': np.std(tv_distances) if tv_distances else np.nan,
            'tv_all': tv_distances,
            'ess_mean': np.mean(ess_values) if ess_values else np.nan,
            'ess_std': np.std(ess_values) if ess_values else np.nan,
            'acceptance_mean': np.mean(acceptance_rates) if acceptance_rates else np.nan,
            'num_chains': len(tv_distances)
        }
        
        return result
    
    def run_all_experiments(self):
        """Run all experiment configurations."""
        logger.info("Starting comprehensive experiment run")
        
        # Generate all configurations
        configs = generate_experiment_configs(
            dimensions=self.dimensions,
            basis_types=self.basis_types,
            ratios=self.ratios
        )
        
        # Run experiments
        for i, config in enumerate(configs):
            logger.info(f"Progress: {i+1}/{len(configs)}")
            result = self.run_single_experiment(config)
            self.results.append(result)
            
            # Save intermediate results
            if (i + 1) % 10 == 0:
                self.save_results()
        
        # Final save
        self.save_results()
        logger.info("Experiment run complete")
    
    def save_results(self):
        """Save results to CSV and JSON formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to JSON
        json_path = self.data_dir / f"results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        # Save to CSV
        csv_path = self.data_dir / f"results_{timestamp}.csv"
        if self.results:
            fieldnames = ['dimension', 'basis_type', 'ratio', 'sigma', 'eta',
                         'tv_mean', 'tv_std', 'ess_mean', 'ess_std', 
                         'acceptance_mean', 'num_chains']
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    row = {k: result[k] for k in fieldnames}
                    writer.writerow(row)
        
        logger.info(f"Results saved to {json_path} and {csv_path}")
    
    def plot_tv_vs_ratio(self):
        """Create TV distance vs ratio plots with error bars."""
        logger.info("Creating TV distance plots")
        
        # Group results by dimension
        for dim in self.dimensions:
            plt.figure(figsize=(10, 6))
            
            # Plot each basis type
            for basis_type in self.basis_types:
                # Filter results
                filtered = [r for r in self.results 
                          if r['dimension'] == dim and r['basis_type'] == basis_type]
                
                if not filtered:
                    continue
                
                # Sort by ratio
                filtered.sort(key=lambda x: x['ratio'])
                
                # Extract data
                ratios = [r['ratio'] for r in filtered]
                tv_means = [r['tv_mean'] for r in filtered]
                tv_stds = [r['tv_std'] for r in filtered]
                
                # Plot with error bars
                plt.errorbar(ratios, tv_means, yerr=tv_stds,
                           label=f"{basis_type}", 
                           marker='o', capsize=5, capthick=2)
            
            plt.xlabel(r'$\sigma/\eta_\epsilon(\Lambda)$')
            plt.ylabel('Total Variation Distance')
            plt.title(f'TV Distance vs Ratio ({dim}D)')
            plt.legend()
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = self.plot_dir / f"tv_vs_ratio_{dim}D.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved plot to {plot_path}")
    
    def plot_ess_comparison(self):
        """Create ESS comparison plots."""
        logger.info("Creating ESS comparison plots")
        
        for dim in self.dimensions:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # ESS vs ratio
            for basis_type in self.basis_types:
                filtered = [r for r in self.results 
                          if r['dimension'] == dim and r['basis_type'] == basis_type]
                
                if not filtered:
                    continue
                
                filtered.sort(key=lambda x: x['ratio'])
                
                ratios = [r['ratio'] for r in filtered]
                ess_means = [r['ess_mean'] for r in filtered]
                ess_stds = [r['ess_std'] for r in filtered]
                
                ax1.errorbar(ratios, ess_means, yerr=ess_stds,
                           label=f"{basis_type}", 
                           marker='o', capsize=5)
            
            ax1.set_xlabel(r'$\sigma/\eta_\epsilon(\Lambda)$')
            ax1.set_ylabel('Effective Sample Size')
            ax1.set_title(f'ESS vs Ratio ({dim}D)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Acceptance rate vs ratio
            for basis_type in self.basis_types:
                filtered = [r for r in self.results 
                          if r['dimension'] == dim and r['basis_type'] == basis_type]
                
                if not filtered:
                    continue
                
                filtered.sort(key=lambda x: x['ratio'])
                
                ratios = [r['ratio'] for r in filtered]
                acc_means = [r['acceptance_mean'] for r in filtered]
                
                ax2.plot(ratios, acc_means,
                        label=f"{basis_type}", 
                        marker='o')
            
            ax2.set_xlabel(r'$\sigma/\eta_\epsilon(\Lambda)$')
            ax2.set_ylabel('Acceptance Rate')
            ax2.set_title(f'Acceptance Rate vs Ratio ({dim}D)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.plot_dir / f"ess_comparison_{dim}D.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved plot to {plot_path}")
    
    def generate_summary_report(self):
        """Generate a summary report with key findings."""
        logger.info("Generating summary report")
        
        report = {
            'experiment_date': datetime.now().isoformat(),
            'num_chains': self.num_chains,
            'num_samples': self.num_samples,
            'dimensions': self.dimensions,
            'basis_types': self.basis_types,
            'ratios': self.ratios,
            'key_findings': []
        }
        
        # Analyze results
        for dim in self.dimensions:
            for basis_type in self.basis_types:
                filtered = [r for r in self.results 
                          if r['dimension'] == dim and r['basis_type'] == basis_type]
                
                if not filtered:
                    continue
                
                # Find optimal ratio
                valid_results = [r for r in filtered if not np.isnan(r['tv_mean'])]
                if valid_results:
                    best = min(valid_results, key=lambda x: x['tv_mean'])
                    
                    finding = {
                        'dimension': dim,
                        'basis_type': basis_type,
                        'optimal_ratio': best['ratio'],
                        'optimal_tv': best['tv_mean'],
                        'optimal_ess': best['ess_mean']
                    }
                    report['key_findings'].append(finding)
        
        # Save report
        report_path = self.output_dir / "experiment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Report saved to {report_path}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        logger.info("Starting complete analysis pipeline")
        
        # Run experiments
        self.run_all_experiments()
        
        # Create plots
        self.plot_tv_vs_ratio()
        self.plot_ess_comparison()
        
        # Generate report
        self.generate_summary_report()
        
        logger.info("Analysis complete")


def main():
    """Main entry point for experiment runner."""
    runner = ExperimentRunner()
    runner.run_complete_analysis()


if __name__ == "__main__":
    main()