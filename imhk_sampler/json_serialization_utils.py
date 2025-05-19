#!/usr/bin/env sage -python
"""
JSON serialization utilities for handling NumPy and SageMath data types.
"""

import json
import numpy as np
import logging
from typing import Any, Dict, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling NumPy and SageMath data types."""
    
    def __init__(self, *args, **kwargs):
        # Force allow_nan to False so we can handle NaN/inf ourselves
        kwargs['allow_nan'] = False
        super().__init__(*args, **kwargs)
    
    def encode(self, o):
        """Custom encode to handle NaN and inf values."""
        # First sanitize the data to handle NaN/inf
        sanitized = self._sanitize_floats(o)
        return super().encode(sanitized)
    
    def _sanitize_floats(self, obj):
        """Recursively sanitize float values."""
        if isinstance(obj, float):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return str("inf") if obj > 0 else str("-inf")
            return obj
        elif isinstance(obj, dict):
            return {k: self._sanitize_floats(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_floats(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return self._sanitize_floats(obj.tolist())
        return obj
    
    def default(self, obj):
        """Convert numpy/sage types to JSON-serializable types."""
        try:
            # Handle numpy floats (including NaN and inf)
            if isinstance(obj, (np.floating, np.float64)):
                if np.isnan(obj):
                    return None  # Convert NaN to None
                elif np.isinf(obj):
                    return str("inf") if obj > 0 else str("-inf")
                return float(obj)
            
            # Handle numpy integers
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            
            # Handle numpy arrays
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            
            # Handle numpy bool
            elif isinstance(obj, np.bool_):
                return bool(obj)
            
            # Handle pathlib.Path objects
            elif isinstance(obj, Path):
                return str(obj)
            
            # Handle numpy scalar types
            elif np.isscalar(obj):
                try:
                    return obj.item()
                except:
                    return float(obj)
            
            else:
                # Log warning for unhandled types
                logger.warning(f"Unhandled type in JSON encoder: {type(obj)} - {obj}")
                return super().default(obj)
                
        except Exception as e:
            logger.error(f"Error serializing object {type(obj)}: {e}")
            return str(obj)  # Fallback to string representation


def sanitize_data_for_json(data: Any, path: str = "root") -> Any:
    """
    Recursively convert data types to JSON-compatible formats.
    
    Args:
        data: Data to sanitize
        path: Current path in data structure (for logging)
    
    Returns:
        JSON-compatible data
    """
    try:
        # Handle None
        if data is None:
            return None
            
        # Handle primitives
        elif isinstance(data, (str, int, float, bool)):
            if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
                logger.warning(f"Converting NaN/Inf at {path} to None")
                return None
            return data
            
        # Handle numpy types
        elif isinstance(data, (np.integer, np.int64)):
            return int(data)
            
        elif isinstance(data, (np.floating, np.float64)):
            if np.isnan(data) or np.isinf(data):
                logger.warning(f"Converting NaN/Inf at {path} to None")
                return None
            return float(data)
            
        elif isinstance(data, np.bool_):
            return bool(data)
            
        elif isinstance(data, np.ndarray):
            logger.debug(f"Converting numpy array at {path}")
            return sanitize_data_for_json(data.tolist(), f"{path}[array]")
            
        # Handle collections
        elif isinstance(data, dict):
            return {
                sanitize_data_for_json(k, f"{path}.{k}"): sanitize_data_for_json(v, f"{path}.{k}")
                for k, v in data.items()
            }
            
        elif isinstance(data, (list, tuple)):
            return [sanitize_data_for_json(item, f"{path}[{i}]") 
                   for i, item in enumerate(data)]
            
        # Handle Path objects
        elif isinstance(data, Path):
            return str(data)
            
        else:
            logger.warning(f"Unknown type at {path}: {type(data)} - converting to string")
            return str(data)
            
    except Exception as e:
        logger.error(f"Error sanitizing data at {path}: {e}")
        return str(data)


def save_json_safely(data: Any, filepath: Union[str, Path], **kwargs) -> None:
    """
    Safely save data to JSON file with error handling.
    
    Args:
        data: Data to save
        filepath: Path to save file
        **kwargs: Additional arguments for json.dump
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Sanitize data first
        sanitized_data = sanitize_data_for_json(data)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(sanitized_data, f, cls=NumpyJSONEncoder, indent=2, **kwargs)
            
        logger.info(f"Successfully saved JSON to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        raise


def validate_json_serializable(data: Any, path: str = "root") -> List[str]:
    """
    Validate that data is JSON serializable and return list of problematic paths.
    
    Args:
        data: Data to validate
        path: Current path in data structure
    
    Returns:
        List of paths with non-serializable data
    """
    problems = []
    
    # Known serializable types
    if data is None or isinstance(data, (str, int, float, bool)):
        # Check for NaN/inf in floats
        if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
            # These are handled by our custom encoder
            return problems
        return problems
    
    # Handle numpy types
    elif isinstance(data, (np.integer, np.int64, np.int32)):
        return problems  # These are handled by our encoder
    elif isinstance(data, (np.floating, np.float64, np.float32)):
        return problems  # These are handled by our encoder
    elif isinstance(data, np.bool_):
        return problems  # These are handled by our encoder
    elif isinstance(data, np.ndarray):
        return problems  # These are handled by our encoder
    
    # Handle collections
    elif isinstance(data, dict):
        for key, value in data.items():
            problems.extend(validate_json_serializable(value, f"{path}.{key}"))
    
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            problems.extend(validate_json_serializable(item, f"{path}[{i}]"))
    
    else:
        # Try to serialize with our custom encoder
        try:
            # First check if it's a basic Python type
            json.dumps(data)
        except (TypeError, ValueError):
            # If basic serialization fails, try with our custom encoder
            try:
                json.dumps(data, cls=NumpyJSONEncoder)
            except (TypeError, ValueError) as e:
                problems.append(f"{path}: {type(data)} - {str(e)}")
    
    return problems


def debug_data_structure(data: Any, max_depth: int = 3, current_depth: int = 0) -> None:
    """
    Print debug information about data structure for troubleshooting.
    
    Args:
        data: Data to debug
        max_depth: Maximum depth to traverse
        current_depth: Current depth in traversal
    """
    indent = "  " * current_depth
    
    if current_depth > max_depth:
        print(f"{indent}... (max depth reached)")
        return
        
    if isinstance(data, dict):
        print(f"{indent}dict with {len(data)} keys:")
        for key in list(data.keys())[:5]:  # Show first 5 keys
            print(f"{indent}  {key}: {type(data[key])}")
            if current_depth < max_depth:
                debug_data_structure(data[key], max_depth, current_depth + 1)
                
    elif isinstance(data, (list, tuple)):
        print(f"{indent}list/tuple with {len(data)} items")
        if len(data) > 0:
            print(f"{indent}  First item type: {type(data[0])}")
            if current_depth < max_depth:
                debug_data_structure(data[0], max_depth, current_depth + 1)
                
    else:
        print(f"{indent}Type: {type(data)}, Value: {str(data)[:50]}...")