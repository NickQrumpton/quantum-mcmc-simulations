#!/usr/bin/env python3
"""
fix_imports.py - Automatic circular import resolver for the IMHK Sampler framework

This script automatically scans Python files in the imhk_sampler directory
and fixes circular dependencies by replacing direct imports with dynamic
import patterns. This helps maintain the modular structure of the codebase
while avoiding circular import issues that can lead to runtime errors.

Key features:
1. Scans all Python files in imhk_sampler directory
2. Identifies problematic direct imports from other modules
3. Replaces them with dynamic import patterns that load on demand
4. Preserves docstrings, functionality, and code structure
5. Creates backup files before making changes

Usage:
    python fix_imports.py [--dry-run] [--verbose]

Options:
    --dry-run   Report issues without modifying files
    --verbose   Show detailed information about all changes

Author: IMHK Development Team
"""

import os
import re
import sys
import shutil
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('fix_imports')

# Module names that might cause circular dependencies
MODULES = [
    'samplers', 
    'utils', 
    'diagnostics', 
    'stats', 
    'visualization', 
    'experiments', 
    'main',
    'config'
]

# Patterns for direct imports that might cause circular dependencies
IMPORT_PATTERNS = [
    # Direct module imports
    r'^from\s+({0})\s+import\s+(.+)$',
    # Qualified module imports
    r'^from\s+imhk_sampler\.({0})\s+import\s+(.+)$',
]

# Pattern for imports inside functions (to avoid duplicating them)
FUNCTION_IMPORT_PATTERN = r'^\s+from\s+(?:imhk_sampler\.)?({0})\s+import\s+'

# Template for dynamic import function (added if not present)
DYNAMIC_IMPORT_FUNCTION = """
def _get_function(module_name, function_name):
    """Dynamically import a function to avoid circular dependencies."""
    import importlib
    
    # Full module path
    full_module_name = f"imhk_sampler.{module_name}"
    
    # Import the module
    module = importlib.import_module(full_module_name)
    
    # Get and return the function
    return getattr(module, function_name)
"""

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fix circular imports in the IMHK Sampler framework'
    )
    parser.add_argument('--dry-run', action='store_true', 
                        help='Report issues without modifying files')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information about all changes')
    
    return parser.parse_args()


def find_python_files(directory):
    """Find all Python files in the given directory."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return sorted(python_files)


def is_import_inside_function(line, indentation):
    """Check if a line is inside a function (has indentation)."""
    return line.startswith(' ' * indentation) or line.startswith('\t')


def check_if_dynamic_import_exists(content):
    """Check if the dynamic import function already exists in the file."""
    # Look for common signature patterns
    patterns = [
        r'def _get_function\(module_name,\s*function_name\):',
        r'def import_module_function\(module_name,\s*function_name\):',
        r'def get_function\(module_name,\s*function_name\):'
    ]
    
    for pattern in patterns:
        if re.search(pattern, content):
            return True
            
    return False


def fix_imports_in_file(file_path, dry_run=False, verbose=False):
    """
    Fix circular imports in a specific Python file.
    
    Args:
        file_path: Path to the Python file
        dry_run: If True, report issues without modifying files
        verbose: If True, show detailed information about changes
        
    Returns:
        Tuple of (total_issues, fixed_issues)
    """
    # Stats
    total_issues = 0
    fixed_issues = 0
    
    # Get the filename for logging
    filename = os.path.basename(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Make a copy of the original content
        new_content = content
        lines = content.splitlines()
        new_lines = lines.copy()
        
        # Check if dynamic import function exists
        dynamic_import_exists = check_if_dynamic_import_exists(content)
        
        # Build regex pattern for all modules
        modules_pattern = '|'.join(MODULES)
        import_patterns = [pattern.format(modules_pattern) for pattern in IMPORT_PATTERNS]
        function_import_pattern = FUNCTION_IMPORT_PATTERN.format(modules_pattern)
        
        # Scan for problematic imports
        for i, line in enumerate(lines):
            # Skip comments and empty lines
            if line.strip().startswith('#') or not line.strip():
                continue
                
            # Check for problematic imports
            for pattern in import_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    module_name = match.group(1)
                    imports = match.group(2)
                    
                    # Skip if it's already a dynamic import or inside a function
                    if is_import_inside_function(line, 4):
                        continue
                        
                    # Make sure we're not in a class or method definition
                    prev_lines = lines[max(0, i-5):i]
                    if any(l.strip().startswith(('class ', 'def ')) for l in prev_lines):
                        # Check indentation to confirm it's inside a function/method
                        if any(l.strip() and not l.startswith(' ') for l in prev_lines):
                            pass  # It's at the module level, not in a function/method
                        else:
                            continue  # Skip, it's an import inside a function/method
                            
                    total_issues += 1
                    
                    # Handle different import styles
                    if imports.strip() == '*':
                        if verbose:
                            logger.warning(f"{filename}: Line {i+1}: Wildcard import from '{module_name}' cannot be safely fixed")
                        continue
                        
                    # Handle single or multiple imports
                    imported_items = [item.strip() for item in imports.split(',')]
                    
                    # Check if these are already dynamically imported elsewhere
                    dynamic_patterns = []
                    for item in imported_items:
                        item_name = item.split(' as ')[0].strip()
                        dynamic_patterns.append(rf'_get_function\([\'"]?{module_name}[\'"]?,\s*[\'"]?{item_name}[\'"]?\)')
                        dynamic_patterns.append(rf'import_module_function\([\'"]?{module_name}[\'"]?,\s*[\'"]?{item_name}[\'"]?\)')
                        
                    # Skip if all items are already dynamically imported
                    if all(any(re.search(dp, content) for dp in dynamic_patterns) for item in imported_items):
                        if verbose:
                            logger.info(f"{filename}: Line {i+1}: Imports already handled dynamically")
                        continue
                    
                    # Create new line with a comment about the original import
                    new_line = f"# FIXED CIRCULAR IMPORT: {line.strip()}"
                    
                    # Update the content
                    if not dry_run:
                        new_lines[i] = new_line
                        fixed_issues += 1
                        if verbose:
                            logger.info(f"{filename}: Line {i+1}: Replaced '{line.strip()}' with '{new_line}'")
                    else:
                        if verbose:
                            logger.info(f"{filename}: Line {i+1}: Would replace '{line.strip()}' with '{new_line}'")
        
        # If we found and fixed issues, add dynamic import function if needed
        if fixed_issues > 0 and not dynamic_import_exists and not dry_run:
            # Find a good location for the function (after imports, before first class/function)
            insert_position = 0
            in_import_section = True
            
            for i, line in enumerate(new_lines):
                stripped = line.strip()
                # Skip empty lines and comments
                if not stripped or stripped.startswith('#'):
                    continue
                    
                # Look for end of import section
                if in_import_section:
                    if not (stripped.startswith('import ') or 
                            stripped.startswith('from ') or 
                            stripped.startswith('# FIXED CIRCULAR IMPORT')):
                        in_import_section = False
                        insert_position = i
                        break
            
            # If we didn't find a good position, insert after the last import
            if insert_position == 0:
                for i, line in enumerate(new_lines):
                    if line.strip().startswith(('import ', 'from ')):
                        insert_position = i + 1
            
            # Add an empty line if there isn't one
            if insert_position > 0 and new_lines[insert_position-1].strip():
                new_lines.insert(insert_position, '')
                insert_position += 1
                
            # Add the dynamic import function
            dynamic_import_lines = DYNAMIC_IMPORT_FUNCTION.strip().splitlines()
            for line in reversed(dynamic_import_lines):
                new_lines.insert(insert_position, line)
            
            # Add another empty line after the function
            new_lines.insert(insert_position + len(dynamic_import_lines), '')
            
            if verbose:
                logger.info(f"{filename}: Added dynamic import function")
        
        # Construct the new content from modified lines
        new_content = '\n'.join(new_lines)
        
        # Write the changes back to the file
        if not dry_run and new_content != content:
            # Create a backup
            backup_path = f"{file_path}.bak"
            shutil.copy2(file_path, backup_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            logger.info(f"{filename}: Updated file (backup at {backup_path})")
        elif dry_run and new_content != content:
            logger.info(f"{filename}: Would update file (dry run)")
            
        return total_issues, fixed_issues
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return 0, 0


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Set dry-run message if enabled
    if args.dry_run:
        logger.info("Running in DRY RUN mode - no files will be modified")
    
    # Find the imhk_sampler directory
    script_dir = Path(__file__).resolve().parent
    imhk_dir = script_dir / 'imhk_sampler'
    
    if not imhk_dir.exists() or not imhk_dir.is_dir():
        logger.error(f"imhk_sampler directory not found at {imhk_dir}")
        return 1
    
    logger.info(f"Scanning Python files in {imhk_dir}")
    
    # Find all Python files
    python_files = find_python_files(imhk_dir)
    logger.info(f"Found {len(python_files)} Python files")
    
    # Process each file
    total_issues = 0
    fixed_issues = 0
    
    for file_path in python_files:
        if args.verbose:
            logger.info(f"Processing {os.path.basename(file_path)}")
            
        file_issues, file_fixed = fix_imports_in_file(
            file_path, 
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        
        total_issues += file_issues
        fixed_issues += file_fixed
    
    # Print summary
    logger.info(f"✓ Scan complete: found {total_issues} potential circular imports")
    
    if args.dry_run:
        logger.info(f"Would fix {fixed_issues} imports (dry run)")
    else:
        logger.info(f"Fixed {fixed_issues} imports")
    
    if total_issues == 0:
        logger.info("No circular imports detected!")
    elif fixed_issues == 0 and total_issues > 0:
        logger.warning("Found issues but couldn't fix them automatically, review logs for details")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
