# IMHK Sampler Setup Instructions

This document provides detailed instructions for setting up and running the IMHK Sampler framework.

## Requirements

The IMHK Sampler has the following dependencies:

- Python 3.7 or higher
- NumPy, SciPy, Matplotlib (core scientific libraries)
- SageMath (for lattice operations)
- Seaborn, tqdm (for visualization and progress display)

## Installation Guide

### Option 1: Automated Setup (Recommended)

1. Run the setup environment script:
   ```
   python setup_environment.py --install
   ```
   
   This will:
   - Check all required dependencies
   - Install missing Python packages
   - Verify the SageMath installation
   - Create necessary directories

2. Install SageMath if not already installed:
   ```
   # Using conda (recommended)
   conda install -c conda-forge sagemath
   
   # OR using pip (limited functionality)
   pip install sagemath
   ```

3. Verify the installation by running the minimal test suite:
   ```
   python imhk_sampler/test_minimal.py
   ```

### Option 2: Quick Run

Use the provided wrapper script to run experiments:
```
python run_imhk.py basic    # Run a basic 2D example
python run_imhk.py test     # Run the minimal test suite
python run_imhk.py validate # Validate the framework functionality
python run_imhk.py sweep    # Run a parameter sweep
```

The wrapper script will also check dependencies and provide installation instructions if needed.

### Option 3: Manual Setup

1. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

2. Install SageMath separately as it may not be available through pip:
   ```
   # For macOS with Homebrew
   brew install sage
   
   # For conda environments
   conda install -c conda-forge sagemath
   
   # For Debian/Ubuntu
   sudo apt-get install sagemath
   ```

3. Create necessary directories:
   ```
   mkdir -p results/plots results/logs data
   ```

## Troubleshooting

### SageMath Installation Issues

If you encounter issues with SageMath:

1. Visit the official installation guide: https://doc.sagemath.org/html/en/installation/index.html
2. For conda environments, try: `conda install -c conda-forge sage`
3. You can run the directory setup test without SageMath: `python imhk_sampler/test_minimal.py`

### Import Errors

If you encounter circular import errors:

1. Run the fix_imports.py script:
   ```
   python imhk_sampler/fix_imports.py
   ```
   
   This will automatically detect and fix circular dependencies in the codebase.

### Missing Results Directory

If plots or logs are not being generated:

1. Create the directories manually:
   ```
   mkdir -p results/plots results/logs data
   ```

2. Ensure you have write permissions to these directories.

## Development Workflow

1. Setup your environment as described above
2. Make changes to the codebase
3. Run the minimal test suite: `python imhk_sampler/test_minimal.py`
4. Run specific experiments: `python run_imhk.py [experiment_type]`
5. If needed, use `python imhk_sampler/fix_imports.py` to fix any circular dependencies

## Contact

If you encounter any issues with the setup or have questions, please contact the Quantum MCMC Research Team.