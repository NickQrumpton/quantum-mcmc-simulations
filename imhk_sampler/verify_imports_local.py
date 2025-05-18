"""
Local import verification for IMHK sampler package.

This script tests all modules for import errors when running from the package directory.
"""

import sys
import importlib
import traceback
from pathlib import Path
from datetime import datetime
import logging

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('import_verification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ImportVerifier:
    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []
        
    def verify_module(self, module_name):
        """Verify that a module can be imported without errors."""
        logger.info(f"Testing import: {module_name}")
        
        try:
            module = importlib.import_module(module_name)
            self.results[module_name] = {
                'status': 'SUCCESS',
                'message': 'Module imported successfully',
                'functions': dir(module)
            }
            logger.info(f"✓ {module_name} imported successfully")
            return True
            
        except ImportError as e:
            self.results[module_name] = {
                'status': 'IMPORT_ERROR',
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            self.errors.append((module_name, str(e)))
            logger.error(f"✗ {module_name}: ImportError - {e}")
            return False
            
        except Exception as e:
            self.results[module_name] = {
                'status': 'OTHER_ERROR',
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            self.errors.append((module_name, str(e)))
            logger.error(f"✗ {module_name}: {type(e).__name__} - {e}")
            return False
    
    def verify_test_modules(self):
        """Verify test modules can import correctly."""
        test_dir = Path(__file__).parent / 'tests'
        
        if not test_dir.exists():
            logger.warning("Tests directory not found")
            return
        
        logger.info("\nVerifying test modules...")
        
        # Add tests directory to path temporarily
        sys.path.insert(0, str(test_dir))
        
        try:
            for test_file in test_dir.glob('test_*.py'):
                module_name = test_file.stem
                self.verify_module(module_name)
        finally:
            sys.path.remove(str(test_dir))
    
    def generate_report(self):
        """Generate a comprehensive diagnostic report."""
        report_path = Path('diagnostics_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("IMHK Sampler Import Diagnostic Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            total_modules = len(self.results)
            successful = sum(1 for r in self.results.values() if r['status'] == 'SUCCESS')
            f.write(f"Total modules tested: {total_modules}\n")
            f.write(f"Successful imports: {successful}\n")
            f.write(f"Failed imports: {total_modules - successful}\n\n")
            
            # Successful imports
            f.write("SUCCESSFUL IMPORTS\n")
            f.write("-" * 20 + "\n")
            for module, result in self.results.items():
                if result['status'] == 'SUCCESS':
                    f.write(f"✓ {module}\n")
            f.write("\n")
            
            # Failed imports
            if self.errors:
                f.write("FAILED IMPORTS\n")
                f.write("-" * 20 + "\n")
                for module, error in self.errors:
                    f.write(f"✗ {module}: {error}\n")
                f.write("\n")
            
            # Detailed error information
            if self.errors:
                f.write("DETAILED ERROR INFORMATION\n")
                f.write("-" * 20 + "\n")
                for module, result in self.results.items():
                    if result['status'] != 'SUCCESS':
                        f.write(f"\nModule: {module}\n")
                        f.write(f"Status: {result['status']}\n")
                        f.write(f"Error: {result['message']}\n")
                        f.write("Traceback:\n")
                        f.write(result['traceback'])
                        f.write("\n" + "-" * 40 + "\n")
        
        logger.info(f"\nDiagnostic report generated: {report_path}")
        return report_path


def main():
    """Run the import verification process."""
    verifier = ImportVerifier()
    
    # Core modules to test
    core_modules = [
        'utils',
        'samplers',
        'stats',
        'diagnostics',
        'visualization',
        'experiments',
        'parameter_config',
        'experiments.report'
    ]
    
    logger.info("Starting local import verification for IMHK sampler...\n")
    
    # Test core modules
    logger.info("Testing core modules...")
    for module in core_modules:
        verifier.verify_module(module)
    
    # Verify test modules
    verifier.verify_test_modules()
    
    # Generate report
    report_path = verifier.generate_report()
    
    # Print summary
    print("\n" + "=" * 50)
    print("VERIFICATION COMPLETE")
    print("=" * 50)
    
    if verifier.errors:
        print(f"\nFound {len(verifier.errors)} import errors:")
        for module, error in verifier.errors:
            print(f"  ✗ {module}: {error}")
        print(f"\nSee {report_path} for detailed information")
        return 1
    else:
        print("\n✓ All imports successful!")
        print(f"\nFull report: {report_path}")
        return 0


if __name__ == "__main__":
    sys.exit(main())