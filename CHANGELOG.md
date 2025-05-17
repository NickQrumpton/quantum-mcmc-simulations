# Changelog

## [1.0.0] - 2025-05-17

### Added
- Created `setup_environment.py` script for dependency management and environment setup
- Added `run_imhk.py` wrapper for user-friendly execution of various experiments
- Created `generate_simulation_results.py` for SageMath-independent results generation
- Added comprehensive setup instructions in SETUP_INSTRUCTIONS.md
- Added detailed README.md with usage examples and project structure
- Created publishable results in results/publication directory
- Added research summary with key findings and visualizations

### Fixed
- Resolved circular dependency issues in IMHK sampler modules
- Fixed syntax errors in fix_imports.py script
- Improved test_minimal.py to handle missing SageMath installation gracefully
- Fixed directory paths in main.py's run_basic_example function
- Updated results directory structure for better organization

### Changed
- Reorganized project structure for better modularity
- Updated fix_imports.py to detect imports correctly
- Improved error handling and diagnostic messages throughout codebase
- Enhanced visualization styling for publication-quality plots
- Improved package initialization with dynamic imports
- Added SageMath dependency checks throughout codebase

### Documentation
- Added detailed installation and setup instructions
- Created comprehensive API documentation for core functions
- Added algorithm selection guidelines based on performance results
- Included example usage in README.md
- Added research summary with key findings