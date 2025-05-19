#!/usr/bin/env sage -python
"""
Patched generate_all_results with proper None handling for formatting.
"""

import logging
import time
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)


def sanitize_value_for_format(value, default=0.0):
    """Ensure value is not None and can be formatted."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def generate_publication_report_patched(self, df):
    """Generate comprehensive publication report with proper None handling."""
    logger.info("Generating publication report (PATCHED)")
    
    report = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'version': '3.0-patched',
            'generator': 'fixed_publication_results_v3_patched.py'
        },
        'summary': {
            'total_experiments': len(df),
            'successful_experiments': len(df[df['error'].isna()]),
            'failed_experiments': len(df[df['error'].notna()]),
            'lattice_types': list(df['basis_type'].unique()),
            'dimensions_tested': sorted(df['dimension'].unique().tolist())
        },
        'metrics_by_type': {},
        'data_completeness': {
            'experiments_with_tv_distance': len(df[df['tv_distance'].notna()]),
            'experiments_with_ess': len(df[df['ess'] > 0]),
            'experiments_with_acceptance_rate': len(df[df['imhk_acceptance_rate'] > 0]),
            'experiments_with_speedup': len(df[df['speedup'] > 0])
        },
        'files_generated': self.metrics_tracker['files_generated'],
        'missing_files': self.metrics_tracker.get('missing_files', [])
    }
    
    # Calculate metrics for each lattice type with proper None handling
    for basis_type in df['basis_type'].unique():
        type_data = df[df['basis_type'] == basis_type]
        
        # Calculate averages with None handling
        avg_acceptance = sanitize_value_for_format(type_data['imhk_acceptance_rate'].mean())
        avg_tv = None
        if type_data['tv_distance'].notna().any():
            avg_tv = sanitize_value_for_format(type_data['tv_distance'].mean())
        avg_ess = sanitize_value_for_format(type_data['ess'].mean())
        avg_speedup = sanitize_value_for_format(type_data['speedup'].mean())
        
        report['metrics_by_type'][basis_type] = {
            'total_experiments': len(type_data),
            'dimensions': sorted(type_data['dimension'].unique().tolist()),
            'average_acceptance_rate': float(avg_acceptance),
            'average_tv_distance': float(avg_tv) if avg_tv is not None else None,
            'average_ess': float(avg_ess),
            'average_speedup': float(avg_speedup),
            'experiments_with_tv': int(type_data['tv_distance'].notna().sum()),
            'experiments_with_ess': int((type_data['ess'] > 0).sum())
        }
    
    # Save report
    report_path = self.output_dir / 'publication_report.json'
    from json_serialization_utils import save_json_safely
    save_json_safely(report, report_path)
    self.metrics_tracker['files_generated'].append(str(report_path))
    
    # Create text summary with proper formatting
    summary_path = self.output_dir / 'summary_report.txt'
    with open(summary_path, 'w') as f:
        f.write("IMHK SAMPLER PUBLICATION RESULTS SUMMARY V3-PATCHED\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {report['metadata']['timestamp']}\n")
        f.write(f"Version: {report['metadata']['version']}\n\n")
        
        f.write("EXPERIMENT SUMMARY:\n")
        f.write(f"  Total experiments: {report['summary']['total_experiments']}\n")
        f.write(f"  Successful: {report['summary']['successful_experiments']}\n")
        f.write(f"  Failed: {report['summary']['failed_experiments']}\n")
        f.write(f"  Lattice types: {', '.join(report['summary']['lattice_types'])}\n")
        f.write(f"  Dimensions: {report['summary']['dimensions_tested']}\n\n")
        
        f.write("METRICS BY LATTICE TYPE:\n")
        for basis_type, metrics in report['metrics_by_type'].items():
            f.write(f"\n{basis_type}:\n")
            f.write(f"  Experiments: {metrics['total_experiments']}\n")
            f.write(f"  Dimensions: {metrics['dimensions']}\n")
            
            # Format with None handling
            avg_acc = sanitize_value_for_format(metrics['average_acceptance_rate'])
            f.write(f"  Avg acceptance rate: {avg_acc:.4f}\n")
            
            if metrics['average_tv_distance'] is not None:
                avg_tv = sanitize_value_for_format(metrics['average_tv_distance'])
                f.write(f"  Avg TV distance: {avg_tv:.6f}\n")
            else:
                f.write(f"  Avg TV distance: N/A\n")
            
            avg_ess = sanitize_value_for_format(metrics['average_ess'])
            avg_speedup = sanitize_value_for_format(metrics['average_speedup'])
            
            f.write(f"  Avg ESS: {avg_ess:.2f}\n")
            f.write(f"  Avg speedup: {avg_speedup:.2f}\n")
            f.write(f"  With TV distance: {metrics['experiments_with_tv']}\n")
            f.write(f"  With ESS: {metrics['experiments_with_ess']}\n")
        
        f.write("\n\nFILES GENERATED:\n")
        for file_path in sorted(report['files_generated']):
            f.write(f"  {file_path}\n")
        
        if report.get('missing_files'):
            f.write("\n\nMISSING FILES:\n")
            for file_path in report['missing_files']:
                f.write(f"  {file_path}\n")
    
    self.metrics_tracker['files_generated'].append(str(summary_path))
    
    return report


def generate_all_results_patched(self):
    """Generate all publication results with comprehensive error handling."""
    logger.info("Starting comprehensive publication results generation (PATCHED)")
    
    try:
        # Import the patched run_comprehensive_experiments
        from run_comprehensive_experiments_patched import run_comprehensive_experiments_patched
        
        # Run experiments with patched version
        df = run_comprehensive_experiments_patched(self)
        
        # Create figures
        self.create_all_figures(df)
        
        # Create tables
        self.create_all_tables(df)
        
        # Generate report with patched version
        report = generate_publication_report_patched(self, df)
        
        logger.info("\n" + "="*60)
        logger.info("PUBLICATION RESULTS GENERATION COMPLETE (PATCHED)")
        logger.info("="*60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Total files generated: {len(self.metrics_tracker['files_generated'])}")
        
        # List all generated files
        logger.info("\nGenerated files:")
        for file_path in sorted(self.metrics_tracker['files_generated']):
            logger.info(f"  ✓ {file_path}")
        
        return report
        
    except Exception as e:
        logger.error(f"Fatal error in result generation: {e}")
        logger.debug(traceback.format_exc())
        raise