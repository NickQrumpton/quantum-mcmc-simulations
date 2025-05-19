# Changelog

All notable changes to the IMHK Sampler project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-05-19

### Added
- **JSON Serialization Enhancements**
  - Custom JSON encoder (`NumpyJSONEncoder`) to handle NumPy and SageMath data types
  - JSON serialization utilities module (`json_serialization_utils.py`) with:
    - `sanitize_data_for_json()` for recursive type conversion
    - `save_json_safely()` for robust JSON saving with error handling
    - `validate_json_serializable()` for data validation
    - `debug_data_structure()` for troubleshooting complex data structures
  
- **Comprehensive Unit Tests**
  - Created `test_generate_publication_results.py` with tests for:
    - JSON serialization for all data types (int, float, list, dict, numpy.int64, numpy.float64, numpy.ndarray)
    - Proper error handling for unsupported data types
    - Specific tests for Q-ary and NTRU lattice results
    - Pandas DataFrame serialization
    - Report generation validation

### Changed
- Updated `generate_publication_results.py` to use safe JSON serialization
- Enhanced logging to provide detailed information about serialization issues
- Improved data type handling in report generation
- Modified JSON saving to use custom encoder throughout

### Fixed
- **JSON Serialization Errors**
  - Fixed serialization errors for numpy.int64, numpy.float64, and numpy.ndarray types
  - Proper handling of NaN and infinity values in JSON output
  - Type conversion issues in pandas DataFrame to JSON conversion
  - Path object serialization issues
  
- **Data Type Handling**
  - Ensured all numeric types are converted to native Python types
  - Fixed grouped data aggregation type issues
  - Resolved pandas Series to dict conversion problems

### Security
- Added validation for JSON serializable data to prevent runtime errors
- Implemented safe error handling for data conversion

## [1.0-publication] - 2024-01-XX

### Added
- `run_full_tv_results.py`: Comprehensive TV distance comparison script
  - Tests dimensions: 2, 4, 8, 16, 32, 64
  - Basis types: identity, skewed, ill-conditioned
  - σ/η ratios from 0.5 to 8.0
  - Generates 10,000 samples per experiment
  - Creates publication-quality plots and detailed analysis

### Fixed
- **ESS/Autocorrelation bug**: Fixed array truth value ambiguity errors
  - Added explicit .size and .any() checks in diagnostics.py
  - Resolved issues with 1D vs 2D array handling
  
- **sklearn dependencies removed**: Eliminated all scikit-learn requirements
  - Visualization now uses pure matplotlib/numpy
  - Added `USE_SKLEARN = False` flag in experiments.py
  
- **Sampler function signatures**: Corrected parameter names for consistency
  - imhk_sampler now uses: B=, sigma=, num_samples=, center=
  - Ensured consistency with Klein sampler interface
  
- **Performance optimizations**: Prevented timeouts in TV distance calculations
  - Fixed field type consistency in stats.py (RDF throughout)
  - Resolved "Point not in lattice" errors with proper type conversion
  
- **End-to-end reliability**: Created comprehensive smoke test
  - `run_minimal_test.py` validates all components
  - Tests all basis types and metrics
  - Completes in under 10 seconds

### Changed
- Merged all fixed files into original modules
- Updated README.md with Quick Smoke Test and Full Publication Run sections
- Streamlined directory structure by removing legacy _fixed files

### Improved
- Documentation clarity in README.md
- Error messages and logging throughout codebase
- Type consistency in SageMath field operations

## [0.9.0] - 2024-01-XX

### Added
- Initial implementation of IMHK sampler
- Klein sampler for baseline comparison
- Comprehensive experiment framework
- Publication-quality visualization tools
- Cryptographic parameter configurations

### Known Issues (Fixed in v1.0)
- Array comparison errors in diagnostics
- sklearn dependencies causing import issues
- Inconsistent function signatures
- TV distance calculation failures
- Timeout issues with large experiments