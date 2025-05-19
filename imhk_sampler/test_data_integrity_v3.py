#!/usr/bin/env sage -python
"""
Enhanced data integrity test v3 with comprehensive logging and validation.
"""

import unittest
import pandas as pd
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import sys
import re
import datetime

# Configure enhanced logging
def setup_test_logging():
    """Set up comprehensive logging for tests."""
    log_dir = Path('fixed_publication_results/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
    )
    
    # File handler
    log_file = log_dir / f'test_data_integrity_v3_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    
    # Configure logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_test_logging()


class TestDataIntegrityV3(unittest.TestCase):
    """Enhanced data integrity tests with comprehensive validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.results_dir = Path('fixed_publication_results')
        logger.info(f"Testing results directory: {self.results_dir}")
        
        # Define all expected files with full paths
        self.expected_files = {
            'data': [
                self.results_dir / 'data' / 'all_results.csv',
                self.results_dir / 'data' / 'all_results.json',
            ],
            'reports': [
                self.results_dir / 'publication_report.json',
                self.results_dir / 'summary_report.txt',
            ],
            'tables': [
                self.results_dir / 'tables' / 'table1_summary_statistics.csv',
                self.results_dir / 'tables' / 'table1_summary_statistics.tex',
                self.results_dir / 'tables' / 'table2_optimal_parameters.csv',
                self.results_dir / 'tables' / 'table2_optimal_parameters.tex',
            ],
            'figures': [
                self.results_dir / 'figures' / 'fig1_tv_distance_comparison.pdf',
                self.results_dir / 'figures' / 'fig1_tv_distance_comparison.png',
                self.results_dir / 'figures' / 'fig2_acceptance_rates_heatmap.pdf',
                self.results_dir / 'figures' / 'fig2_acceptance_rates_heatmap.png',
                self.results_dir / 'figures' / 'fig3_performance_analysis.pdf',
                self.results_dir / 'figures' / 'fig3_performance_analysis.png',
            ]
        }
        
        # Expected lattice types and metrics
        self.expected_lattice_types = ['identity', 'q-ary', 'NTRU', 'PrimeCyclotomic']
        self.expected_metrics = [
            'imhk_acceptance_rate', 'tv_distance', 'ess', 
            'ess_per_second', 'speedup', 'imhk_time', 'klein_time'
        ]
        
        # Minimum expected data points
        self.min_data_points = {
            'identity': 15,
            'q-ary': 12,
            'NTRU': 3,
            'PrimeCyclotomic': 3
        }
    
    def test_directory_structure(self):
        """Test that all required directories exist."""
        logger.info("\n=== Testing Directory Structure ===")
        
        required_dirs = [
            self.results_dir,
            self.results_dir / 'data',
            self.results_dir / 'figures',
            self.results_dir / 'tables',
            self.results_dir / 'logs'
        ]
        
        for dir_path in required_dirs:
            with self.subTest(directory=str(dir_path)):
                self.assertTrue(dir_path.exists(), f"Directory missing: {dir_path}")
                logger.info(f"✓ Directory exists: {dir_path}")
    
    def test_all_files_exist(self):
        """Test that all required files exist."""
        logger.info("\n=== Testing File Existence ===")
        
        missing_files = []
        existing_files = []
        
        for category, files in self.expected_files.items():
            logger.info(f"\nChecking {category} files:")
            for file_path in files:
                if file_path.exists():
                    logger.info(f"  ✓ {file_path.name}")
                    existing_files.append(str(file_path))
                else:
                    logger.error(f"  ✗ {file_path.name} - MISSING")
                    missing_files.append(str(file_path))
        
        logger.info(f"\nSummary: {len(existing_files)} files exist, {len(missing_files)} missing")
        
        if missing_files:
            logger.error(f"Missing files: {missing_files}")
        
        self.assertEqual(len(missing_files), 0, 
                        f"Missing {len(missing_files)} files")
    
    def test_csv_data_completeness(self):
        """Test completeness of data in CSV files."""
        logger.info("\n=== Testing CSV Data Completeness ===")
        
        csv_path = self.results_dir / 'data' / 'all_results.csv'
        
        if not csv_path.exists():
            self.fail(f"Critical file missing: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from {csv_path}")
        
        # Check all lattice types are present
        actual_types = set(df['basis_type'].unique())
        missing_types = set(self.expected_lattice_types) - actual_types
        
        if missing_types:
            logger.error(f"Missing lattice types: {missing_types}")
        
        self.assertEqual(len(missing_types), 0, 
                        f"Missing lattice types: {missing_types}")
        logger.info(f"All lattice types present: {actual_types}")
        
        # Check data points per lattice type
        logger.info("\nData points per lattice type:")
        data_summary = {}
        
        for lattice_type in self.expected_lattice_types:
            type_data = df[df['basis_type'] == lattice_type]
            count = len(type_data)
            expected_min = self.min_data_points.get(lattice_type, 1)
            
            data_summary[lattice_type] = {
                'count': count,
                'expected_min': expected_min,
                'status': '✓' if count >= expected_min else '✗'
            }
            
            logger.info(f"  {lattice_type}: {count} experiments "
                       f"(expected ≥ {expected_min}) {data_summary[lattice_type]['status']}")
            
            with self.subTest(lattice_type=lattice_type):
                self.assertGreaterEqual(count, expected_min,
                                      f"{lattice_type} has insufficient data points")
            
            # Check metrics completeness
            self._check_metrics_for_type(type_data, lattice_type)
        
        return data_summary
    
    def _check_metrics_for_type(self, type_data, lattice_type):
        """Check metrics completeness for a specific lattice type."""
        logger.info(f"\n  Metrics for {lattice_type}:")
        
        metrics_summary = {}
        
        for metric in self.expected_metrics:
            if metric not in type_data.columns:
                logger.error(f"    ✗ {metric}: column missing")
                metrics_summary[metric] = {'status': 'missing'}
                continue
            
            null_count = type_data[metric].isnull().sum()
            zero_count = (type_data[metric] == 0).sum()
            valid_count = len(type_data) - null_count
            
            # Calculate percentage
            valid_pct = (valid_count / len(type_data)) * 100 if len(type_data) > 0 else 0
            
            metrics_summary[metric] = {
                'valid': valid_count,
                'zero': zero_count,
                'null': null_count,
                'valid_pct': valid_pct
            }
            
            logger.info(f"    {metric}: {valid_count}/{len(type_data)} valid "
                       f"({valid_pct:.1f}%), {zero_count} zeros, {null_count} nulls")
            
            # Critical metrics should have data
            if metric == 'imhk_acceptance_rate':
                self.assertGreater(valid_count, 0,
                                 f"No acceptance rate data for {lattice_type}")
            elif metric == 'imhk_time':
                self.assertGreater(valid_count, 0,
                                 f"No timing data for {lattice_type}")
        
        return metrics_summary
    
    def test_figure_files_exist_and_valid(self):
        """Test that all figures exist and are valid files."""
        logger.info("\n=== Testing Figure Files ===")
        
        figure_files = self.expected_files['figures']
        figure_status = {}
        
        for fig_path in figure_files:
            fig_name = fig_path.name
            
            if fig_path.exists():
                size = fig_path.stat().st_size
                figure_status[fig_name] = {
                    'exists': True,
                    'size': size,
                    'valid': size > 5000  # Expect at least 5KB for a valid figure
                }
                
                if figure_status[fig_name]['valid']:
                    logger.info(f"  ✓ {fig_name}: {size:,} bytes")
                else:
                    logger.warning(f"  ⚠ {fig_name}: {size:,} bytes (too small)")
                
                with self.subTest(figure=fig_name):
                    self.assertGreater(size, 5000, 
                                     f"{fig_name} is too small ({size} bytes)")
            else:
                figure_status[fig_name] = {
                    'exists': False,
                    'size': 0,
                    'valid': False
                }
                logger.error(f"  ✗ {fig_name}: MISSING")
                
                with self.subTest(figure=fig_name):
                    self.fail(f"Missing figure: {fig_name}")
        
        return figure_status
    
    def test_table_content_validity(self):
        """Test content validity of generated tables."""
        logger.info("\n=== Testing Table Content ===")
        
        table_status = {}
        
        # Test summary statistics table
        summary_csv = self.results_dir / 'tables' / 'table1_summary_statistics.csv'
        if summary_csv.exists():
            try:
                df = pd.read_csv(summary_csv)
                
                # Check if it's multi-index (grouped data)
                if df.columns[0].startswith('Unnamed'):
                    # Read with multi-index
                    df = pd.read_csv(summary_csv, index_col=[0, 1])
                
                table_status['summary_statistics'] = {
                    'exists': True,
                    'rows': len(df),
                    'valid': len(df) > 0
                }
                
                logger.info(f"  ✓ Summary statistics table: {len(df)} rows")
                
                # Check that all lattice types are included
                if hasattr(df.index, 'levels'):
                    types = df.index.get_level_values(0).unique()
                    for expected_type in self.expected_lattice_types:
                        if expected_type not in types:
                            logger.warning(f"    ⚠ Missing type: {expected_type}")
                
            except Exception as e:
                logger.error(f"  ✗ Error reading summary table: {e}")
                table_status['summary_statistics'] = {
                    'exists': True,
                    'rows': 0,
                    'valid': False,
                    'error': str(e)
                }
        else:
            logger.error(f"  ✗ Summary statistics table missing")
            table_status['summary_statistics'] = {'exists': False}
        
        # Test optimal parameters table
        optimal_csv = self.results_dir / 'tables' / 'table2_optimal_parameters.csv'
        if optimal_csv.exists():
            try:
                df = pd.read_csv(optimal_csv)
                
                table_status['optimal_parameters'] = {
                    'exists': True,
                    'rows': len(df),
                    'valid': len(df) > 0
                }
                
                logger.info(f"  ✓ Optimal parameters table: {len(df)} rows")
                
                # Check columns
                expected_cols = ['Lattice Type', 'Dimension', 'Optimal σ/η', 
                               'Acceptance Rate', 'TV Distance', 'ESS']
                
                missing_cols = set(expected_cols) - set(df.columns)
                if missing_cols:
                    logger.warning(f"    ⚠ Missing columns: {missing_cols}")
                
            except Exception as e:
                logger.error(f"  ✗ Error reading optimal parameters table: {e}")
                table_status['optimal_parameters'] = {
                    'exists': True,
                    'rows': 0,
                    'valid': False,
                    'error': str(e)
                }
        else:
            logger.error(f"  ✗ Optimal parameters table missing")
            table_status['optimal_parameters'] = {'exists': False}
        
        return table_status
    
    def test_json_report_structure(self):
        """Test JSON report structure and content."""
        logger.info("\n=== Testing JSON Report Structure ===")
        
        report_path = self.results_dir / 'publication_report.json'
        
        if not report_path.exists():
            self.fail(f"Publication report missing: {report_path}")
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        # Check required sections
        required_sections = ['metadata', 'summary', 'metrics_by_type', 
                           'data_completeness', 'files_generated']
        
        missing_sections = []
        for section in required_sections:
            if section in report:
                logger.info(f"  ✓ Section '{section}' present")
            else:
                logger.error(f"  ✗ Section '{section}' missing")
                missing_sections.append(section)
        
        self.assertEqual(len(missing_sections), 0,
                        f"Missing report sections: {missing_sections}")
        
        # Check metrics by type
        if 'metrics_by_type' in report:
            logger.info("\n  Metrics by type:")
            for lattice_type, metrics in report['metrics_by_type'].items():
                logger.info(f"    {lattice_type}:")
                logger.info(f"      Experiments: {metrics.get('total_experiments', 0)}")
                logger.info(f"      Avg acceptance: {metrics.get('average_acceptance_rate', 0):.4f}")
                logger.info(f"      Avg TV: {metrics.get('average_tv_distance', 'N/A')}")
                logger.info(f"      Avg ESS: {metrics.get('average_ess', 0):.2f}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive integrity report."""
        report_path = self.results_dir / 'data_integrity_report_v3.txt'
        
        logger.info(f"\nGenerating comprehensive report: {report_path}")
        
        with open(report_path, 'w') as f:
            f.write("DATA INTEGRITY REPORT V3\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.datetime.now()}\n")
            f.write(f"Test directory: {self.results_dir}\n\n")
            
            # Check directories
            f.write("DIRECTORY STRUCTURE:\n")
            for category, files in self.expected_files.items():
                dir_path = Path(files[0]).parent
                exists = dir_path.exists()
                f.write(f"  {category}: {dir_path} - {'EXISTS' if exists else 'MISSING'}\n")
            
            # Check files
            f.write("\n\nFILE EXISTENCE:\n")
            for category, files in self.expected_files.items():
                f.write(f"\n{category.upper()}:\n")
                for file_path in files:
                    exists = file_path.exists()
                    size = file_path.stat().st_size if exists else 0
                    f.write(f"  {file_path.name}: {'EXISTS' if exists else 'MISSING'}")
                    if exists:
                        f.write(f" ({size:,} bytes)")
                    f.write("\n")
            
            # Data summary
            csv_path = self.results_dir / 'data' / 'all_results.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                f.write("\n\nDATA SUMMARY:\n")
                f.write(f"Total experiments: {len(df)}\n")
                f.write(f"Lattice types: {list(df['basis_type'].unique())}\n")
                f.write(f"Dimensions: {sorted(df['dimension'].unique())}\n")
                
                f.write("\n\nMETRICS BY LATTICE TYPE:\n")
                for lattice_type in df['basis_type'].unique():
                    type_data = df[df['basis_type'] == lattice_type]
                    f.write(f"\n{lattice_type.upper()}:\n")
                    f.write(f"  Experiments: {len(type_data)}\n")
                    f.write(f"  Dimensions: {sorted(type_data['dimension'].unique())}\n")
                    
                    for metric in self.expected_metrics:
                        if metric in type_data.columns:
                            valid = type_data[metric].notna().sum()
                            pct = (valid / len(type_data)) * 100
                            avg = type_data[metric].mean() if valid > 0 else 0
                            f.write(f"  {metric}: {valid}/{len(type_data)} "
                                   f"({pct:.1f}% valid, avg={avg:.4f})\n")
            else:
                f.write("\n\nERROR: Main results CSV not found!\n")
            
            # Overall assessment
            all_files_exist = all(
                file_path.exists() 
                for files in self.expected_files.values() 
                for file_path in files
            )
            
            f.write(f"\n\nOVERALL ASSESSMENT: {'COMPLETE' if all_files_exist else 'INCOMPLETE'}\n")
            
            if not all_files_exist:
                f.write("\nMissing files:\n")
                for category, files in self.expected_files.items():
                    for file_path in files:
                        if not file_path.exists():
                            f.write(f"  - {file_path}\n")
        
        logger.info(f"Report saved to: {report_path}")
        return report_path


def run_comprehensive_tests():
    """Run all tests and generate comprehensive report."""
    logger.info("Starting comprehensive data integrity tests V3")
    logger.info("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataIntegrityV3)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generate comprehensive report
    test = TestDataIntegrityV3()
    test.setUp()
    report_path = test.generate_comprehensive_report()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Success: {result.wasSuccessful()}")
    logger.info(f"Report saved: {report_path}")
    
    # Print report content
    logger.info("\nREPORT CONTENT:")
    with open(report_path, 'r') as f:
        logger.info(f.read())
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)