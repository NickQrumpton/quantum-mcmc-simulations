# Changelog

All notable changes to the IMHK Sampler project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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