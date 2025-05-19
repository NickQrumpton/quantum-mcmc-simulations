#!/usr/bin/env sage -python
"""
Unit tests for generate_publication_results.py
Tests JSON serialization, data type handling, and report generation.
"""

import unittest
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from json_serialization_utils import (
    NumpyJSONEncoder,
    sanitize_data_for_json,
    save_json_safely,
    validate_json_serializable
)
from generate_publication_results import PublicationResultsGenerator


class TestJSONSerialization(unittest.TestCase):
    """Test JSON serialization utilities for numpy and pandas types."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = {
            'int': 42,
            'float': 3.14159,
            'str': 'test',
            'bool': True,
            'numpy_int64': np.int64(42),
            'numpy_float64': np.float64(3.14159),
            'numpy_array': np.array([1, 2, 3]),
            'numpy_2d_array': np.array([[1, 2], [3, 4]]),
            'nan_value': float('nan'),  # Regular Python NaN
            'numpy_nan': np.nan,  # Numpy NaN
            'inf_value': float('inf'),  # Regular Python inf
            'numpy_inf': np.inf,  # Numpy inf
            'none': None,
            'list': [1, 2, 3],
            'dict': {'key': 'value'},
            'nested': {
                'array': np.array([1.0, 2.0, 3.0]),
                'int': np.int64(100)
            }
        }
        
    def test_numpy_json_encoder(self):
        """Test custom JSON encoder handles numpy types."""
        # Test with numpy types
        encoded = json.dumps(self.test_data, cls=NumpyJSONEncoder)
        decoded = json.loads(encoded)
        
        # Check conversions
        self.assertEqual(decoded['numpy_int64'], 42)
        self.assertEqual(decoded['numpy_float64'], 3.14159)
        self.assertEqual(decoded['numpy_array'], [1, 2, 3])
        self.assertEqual(decoded['numpy_2d_array'], [[1, 2], [3, 4]])
        # NaN values should be converted to None
        self.assertIsNone(decoded['nan_value'])
        self.assertIsNone(decoded['numpy_nan'])
        # inf values should be handled as strings
        self.assertEqual(decoded['inf_value'], "inf")
        self.assertEqual(decoded['numpy_inf'], "inf")
        
    def test_sanitize_data_for_json(self):
        """Test data sanitization function."""
        sanitized = sanitize_data_for_json(self.test_data)
        
        # Check that all numpy types are converted
        self.assertIsInstance(sanitized['numpy_int64'], int)
        self.assertIsInstance(sanitized['numpy_float64'], float)
        self.assertIsInstance(sanitized['numpy_array'], list)
        self.assertIsInstance(sanitized['numpy_2d_array'], list)
        
        # Check NaN and inf handling
        self.assertIsNone(sanitized['nan_value'])
        self.assertIsNone(sanitized['numpy_nan'])
        self.assertIsNone(sanitized['inf_value'])
        self.assertIsNone(sanitized['numpy_inf'])
        
    def test_save_json_safely(self):
        """Test safe JSON saving with error handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.json'
            
            # Test saving with numpy types
            save_json_safely(self.test_data, filepath)
            
            # Load and verify
            with open(filepath) as f:
                loaded = json.load(f)
            
            self.assertEqual(loaded['numpy_int64'], 42)
            self.assertEqual(loaded['numpy_array'], [1, 2, 3])
            
    def test_validate_json_serializable(self):
        """Test JSON validation function."""
        # Test valid data
        problems = validate_json_serializable({'key': 'value'})
        self.assertEqual(len(problems), 0)
        
        # Test numpy types that should be valid
        numpy_data = {
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'numpy_array': np.array([1, 2, 3]),
            'nan_value': np.nan,
            'inf_value': np.inf
        }
        problems = validate_json_serializable(numpy_data)
        self.assertEqual(len(problems), 0, "Numpy types should be valid")
        
        # The key test: ensure JSON can be produced
        # This is what matters in practice
        json_str = json.dumps(numpy_data, cls=NumpyJSONEncoder)
        self.assertIsInstance(json_str, str)
        
        # Verify the encoded data
        decoded = json.loads(json_str)
        self.assertEqual(decoded['numpy_int'], 42)
        self.assertEqual(decoded['numpy_float'], 3.14)
        self.assertEqual(decoded['numpy_array'], [1, 2, 3])
        self.assertIsNone(decoded['nan_value'])
        self.assertEqual(decoded['inf_value'], "inf")
        
    def test_pandas_dataframe_serialization(self):
        """Test serialization of pandas DataFrame objects."""
        df = pd.DataFrame({
            'int_col': np.array([1, 2, 3], dtype=np.int64),
            'float_col': np.array([1.1, 2.2, 3.3], dtype=np.float64),
            'str_col': ['a', 'b', 'c']
        })
        
        # Convert to dict for JSON
        data_dict = df.to_dict('records')
        sanitized = sanitize_data_for_json(data_dict)
        
        # Should be serializable
        json_str = json.dumps(sanitized)
        loaded = json.loads(json_str)
        
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0]['int_col'], 1)
        self.assertEqual(loaded[0]['float_col'], 1.1)
        

class TestPublicationResultsGeneration(unittest.TestCase):
    """Test the publication results generation process."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.generator = PublicationResultsGenerator(output_dir=self.test_dir)
        
    def test_result_dataframe_creation(self):
        """Test creation of results dataframe with proper types."""
        # Create mock results
        results = []
        for dim in [8, 16]:
            for basis in ['identity', 'q-ary']:
                result = {
                    'dimension': np.int64(dim),
                    'basis_type': basis,
                    'sigma_ratio': np.float64(1.5),
                    'imhk_acceptance_rate': np.float64(0.75),
                    'tv_distance': np.float64(0.1) if np.random.rand() > 0.5 else np.nan,
                    'ess': np.random.rand() * 100
                }
                results.append(result)
        
        # Test that DataFrame can be created and saved
        df = pd.DataFrame(results)
        
        # Save as JSON using our safe method
        output_file = Path(self.test_dir) / 'test_results.json'
        save_json_safely(results, output_file)
        
        # Verify it can be loaded
        with open(output_file) as f:
            loaded = json.load(f)
        
        self.assertEqual(len(loaded), len(results))
        
    def test_report_generation(self):
        """Test report generation with proper type handling."""
        # Create mock dataframe
        data = {
            'dimension': [8, 16, 8, 16],
            'basis_type': ['identity', 'identity', 'q-ary', 'q-ary'],
            'sigma_ratio': [1.0, 1.5, 2.0, 2.5],
            'imhk_acceptance_rate': [0.5, 0.6, 0.7, 0.8],
            'tv_distance': [0.1, 0.15, np.nan, 0.25],
            'ess': [50, 60, 70, 80]
        }
        df = pd.DataFrame(data)
        
        # Convert types as in the actual code
        lattice_types = [str(lt) for lt in df['basis_type'].unique()]
        dimensions = [int(d) for d in sorted(df['dimension'].unique())]
        
        # Test grouped operations
        acceptance_rates = {}
        for basis, rate in df.groupby('basis_type')['imhk_acceptance_rate'].mean().items():
            acceptance_rates[str(basis)] = float(rate) if pd.notna(rate) else None
            
        report = {
            'total_experiments': int(len(df)),
            'lattice_types': lattice_types,
            'dimensions_tested': dimensions,
            'average_acceptance_rates': acceptance_rates
        }
        
        # Should be JSON serializable
        json_str = json.dumps(report)
        loaded = json.loads(json_str)
        
        self.assertEqual(loaded['total_experiments'], 4)
        self.assertIn('identity', loaded['lattice_types'])
        self.assertIn('q-ary', loaded['lattice_types'])
        

class TestQaryNTRUSerialization(unittest.TestCase):
    """Test serialization specifically for Q-ary and NTRU experiment results."""
    
    def test_cryptographic_lattice_results(self):
        """Test serialization of Q-ary and NTRU specific results."""
        # Simulate results from cryptographic lattice experiments
        crypto_results = []
        
        # Q-ary results
        for dim in [16, 32, 64]:
            result = {
                'basis_type': 'q-ary',
                'dimension': np.int64(dim),
                'q_value': np.int64(2**int(dim/2) + 1),  # Prime modulus
                'sigma_ratio': np.float64(1.5),
                'imhk_acceptance_rate': np.float64(0.95),
                'tv_distance': np.float64(0.05),
                'klein_samples': np.array([1, 2, 3, 4, 5])  # Sample vector
            }
            crypto_results.append(result)
            
        # NTRU results
        for N in [512, 1024]:
            result = {
                'basis_type': 'NTRU',
                'dimension': np.int64(N),
                'q_value': np.int64(12289),  # Falcon parameter
                'poly_degree': np.int64(N),
                'sigma_ratio': np.float64(2.0),
                'imhk_acceptance_rate': np.float64(0.0),  # Expected 0 for structured
                'tv_distance': None  # May be None for high dimensions
            }
            crypto_results.append(result)
            
        # Test serialization
        sanitized = sanitize_data_for_json(crypto_results)
        json_str = json.dumps(sanitized)
        loaded = json.loads(json_str)
        
        # Verify conversions
        self.assertEqual(len(loaded), 5)
        
        # Check Q-ary results
        qary_result = next(r for r in loaded if r['basis_type'] == 'q-ary')
        self.assertIsInstance(qary_result['dimension'], int)
        self.assertIsInstance(qary_result['q_value'], int)
        self.assertIsInstance(qary_result['tv_distance'], float)
        self.assertIsInstance(qary_result['klein_samples'], list)
        
        # Check NTRU results
        ntru_result = next(r for r in loaded if r['basis_type'] == 'NTRU')
        self.assertIsInstance(ntru_result['dimension'], int)
        self.assertEqual(ntru_result['imhk_acceptance_rate'], 0.0)
        self.assertIsNone(ntru_result['tv_distance'])


if __name__ == '__main__':
    # Run the test suite
    unittest.main(verbosity=2)