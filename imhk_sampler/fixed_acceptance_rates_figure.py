#!/usr/bin/env sage -python
"""
Fixed acceptance rates figure generation for ensuring proper heatmap creation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_acceptance_rates_figure_fixed(df, figures_dir, metrics_tracker):
    """Create acceptance rates heatmap with better error handling."""
    logger.info("Creating fixed acceptance rates heatmap")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Ensure all lattice types are included
    all_types = ['identity', 'q-ary', 'NTRU', 'PrimeCyclotomic']
    
    # Create aggregated data for heatmap with better structure
    heatmap_data = []
    
    # First pass: collect all unique sigma ratios
    all_sigma_ratios = sorted(df['sigma_ratio'].unique())
    sigma_ratio_labels = [f'{sr:.2f}' for sr in all_sigma_ratios]
    
    # Create a complete matrix with defaults
    lattice_labels = []
    acceptance_matrix = []
    
    for basis_type in all_types:
        type_data = df[df['basis_type'] == basis_type]
        
        if type_data.empty:
            # Add empty row for missing lattice type
            lattice_labels.append(basis_type)
            acceptance_matrix.append([np.nan] * len(all_sigma_ratios))
            logger.warning(f"No data for {basis_type}, adding empty row")
            continue
        
        if basis_type == 'q-ary':
            # Handle Q-ary by dimension
            dims = sorted(type_data['dimension'].unique())
            for dim in dims:
                dim_data = type_data[type_data['dimension'] == dim]
                label = f'{basis_type} (d={dim})'
                lattice_labels.append(label)
                
                # Create row for this dimension
                row = []
                for sr in all_sigma_ratios:
                    sr_data = dim_data[dim_data['sigma_ratio'] == sr]
                    if not sr_data.empty:
                        mean_rate = sr_data['imhk_acceptance_rate'].mean()
                        row.append(mean_rate)
                    else:
                        row.append(np.nan)
                
                acceptance_matrix.append(row)
        else:
            # Handle other types
            lattice_labels.append(basis_type)
            row = []
            for sr in all_sigma_ratios:
                sr_data = type_data[type_data['sigma_ratio'] == sr]
                if not sr_data.empty:
                    mean_rate = sr_data['imhk_acceptance_rate'].mean()
                    row.append(mean_rate)
                else:
                    row.append(np.nan)
            
            acceptance_matrix.append(row)
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(acceptance_matrix, 
                             index=lattice_labels, 
                             columns=sigma_ratio_labels)
    
    # Remove any rows/columns that are completely NaN
    heatmap_df = heatmap_df.dropna(axis=0, how='all')
    heatmap_df = heatmap_df.dropna(axis=1, how='all')
    
    if heatmap_df.empty:
        logger.error("No valid data for acceptance rates heatmap")
        ax.text(0.5, 0.5, 'No acceptance rate data available',
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
    else:
        # Create heatmap with custom formatting
        sns.heatmap(heatmap_df, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Acceptance Rate'}, 
                   vmin=0, 
                   vmax=1,
                   linewidths=0.5,
                   linecolor='gray',
                   cbar=True,
                   square=False)
        
        # Rotate tick labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        logger.info(f"Heatmap created with shape: {heatmap_df.shape}")
    
    ax.set_xlabel('σ/η Ratio')
    ax.set_ylabel('Lattice Type')
    ax.set_title('IMHK Acceptance Rates Across All Lattice Types', fontsize=16, pad=20)
    
    plt.tight_layout()
    
    # Save in both formats with error handling
    try:
        for ext in ['pdf', 'png']:
            fig_path = figures_dir / f'fig2_acceptance_rates_heatmap.{ext}'
            plt.savefig(fig_path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
            metrics_tracker['files_generated'].append(str(fig_path))
            logger.info(f"Saved acceptance rates figure: {fig_path}")
    except Exception as e:
        logger.error(f"Failed to save acceptance rates figure: {e}")
    finally:
        plt.close(fig)