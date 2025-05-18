"""
Comprehensive import verification for IMHK sampler package.

This script tests all modules for import errors and generates a diagnostic report.
"""

import sys
import importlib
import traceback
from pathlib import Path
from datetime import datetime
import logging

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
        
    def verify_module(self, module_name, parent_path='imhk_sampler'):
        """Verify that a module can be imported without errors."""
        full_name = f"{parent_path}.{module_name}" if parent_path else module_name
        
        logger.info(f"Testing import: {full_name}")
        
        try:
            module = importlib.import_module(full_name)
            self.results[full_name] = {
                'status': 'SUCCESS',
                'message': 'Module imported successfully',
                'functions': dir(module)
            }
            logger.info(f"✓ {full_name} imported successfully")
            return True
            
        except ImportError as e:
            self.results[full_name] = {
                'status': 'IMPORT_ERROR',
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            self.errors.append((full_name, str(e)))
            logger.error(f"✗ {full_name}: ImportError - {e}")
            return False
            
        except Exception as e:
            self.results[full_name] = {
                'status': 'OTHER_ERROR',
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            self.errors.append((full_name, str(e)))
            logger.error(f"✗ {full_name}: {type(e).__name__} - {e}")
            return False
    
    def check_circular_imports(self):
        """Check for potential circular import issues."""
        logger.info("\nChecking for circular imports...")
        
        # Test import order dependencies
        test_orders = [
            ['utils', 'samplers', 'stats', 'diagnostics'],
            ['samplers', 'utils', 'diagnostics', 'stats'],
            ['stats', 'utils', 'samplers', 'diagnostics']
        ]
        
        for order in test_orders:
            logger.info(f"Testing import order: {' -> '.join(order)}")
            
            # Clear modules
            for module in order:
                full_name = f'imhk_sampler.{module}'
                if full_name in sys.modules:
                    del sys.modules[full_name]
            
            # Try importing in this order
            success = True
            for module in order:
                if not self.verify_module(module):
                    success = False
                    break
            
            if success:
                logger.info(f"✓ Import order successful: {' -> '.join(order)}")
            else:
                self.warnings.append(f"Import order failed: {' -> '.join(order)}")
                logger.warning(f"✗ Import order failed: {' -> '.join(order)}")
    
    def verify_test_modules(self):
        """Verify test modules can import correctly."""
        test_dir = Path(__file__).parent / 'tests'
        
        if not test_dir.exists():
            logger.warning("Tests directory not found")
            return
        
        logger.info("\nVerifying test modules...")
        
        for test_file in test_dir.glob('test_*.py'):
            module_name = test_file.stem
            
            # Add tests directory to path temporarily
            sys.path.insert(0, str(test_dir))
            
            try:
                self.verify_module(module_name, parent_path='')
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
            f.write(f"Failed imports: {total_modules - successful}\n")
            f.write(f"Warnings: {len(self.warnings)}\n\n")
            
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
            
            # Warnings
            if self.warnings:
                f.write("\nWARNINGS\n")
                f.write("-" * 20 + "\n")
                for warning in self.warnings:
                    f.write(f"⚠ {warning}\n")
            
            # Recommendations
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            if self.errors:
                f.write("1. Fix import errors in the following order:\n")
                for i, (module, error) in enumerate(self.errors):
                    f.write(f"   {i+1}. {module}: {error}\n")
                f.write("\n2. Common solutions:\n")
                f.write("   - Check for missing dependencies\n")
                f.write("   - Verify correct module paths\n")
                f.write("   - Look for circular import issues\n")
                f.write("   - Ensure __init__.py files are present\n")
            else:
                f.write("All imports are working correctly!\n")
        
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
    
    logger.info("Starting import verification for IMHK sampler...\n")
    
    # Test core modules
    logger.info("Testing core modules...")
    for module in core_modules:
        verifier.verify_module(module)
    
    # Check for circular imports
    verifier.check_circular_imports()
    
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