"""
Configuration module for the IMHK Sampler framework.

This module provides centralized path management and directory structure
for the Independent Metropolis-Hastings-Klein (IMHK) Sampler framework.
It ensures that all necessary directories exist and provides consistent
access to file paths throughout the framework.
"""

from pathlib import Path
import os
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("imhk_config")

class Config:
    """
    Configuration class for the IMHK Sampler framework.
    
    This class manages all path-related operations and ensures that 
    necessary directories are created and accessible throughout the framework.
    All methods are class methods, so no instantiation is required.
    
    Attributes:
        ROOT_DIR (Path): The root project directory
        RESULTS_DIR (Path): Directory for storing all results
        PLOTS_DIR (Path): Directory for storing visualization plots
        LOGS_DIR (Path): Directory for storing log files
        DATA_DIR (Path): Directory for storing input/output data
    """
    
    # Directory paths will be set during initialization
    ROOT_DIR = None
    RESULTS_DIR = None
    PLOTS_DIR = None
    LOGS_DIR = None
    DATA_DIR = None
    INITIALIZED = False
    
    @classmethod
    def initialize(cls, root_dir=None):
        """
        Initialize the configuration and create directory structure.
        
        Args:
            root_dir (Path or str, optional): The root directory for the project.
                If not provided, it will be determined automatically from the 
                location of this file.
        
        Returns:
            bool: True if initialization is successful
        """
        if cls.INITIALIZED:
            logger.debug("Config already initialized")
            return True
            
        try:
            # Determine root directory
            if root_dir is None:
                # Default: two levels up from this file (assuming imhk_sampler/config.py)
                cls.ROOT_DIR = Path(__file__).resolve().parent.parent
            else:
                cls.ROOT_DIR = Path(root_dir).resolve()
            
            # Set directory paths
            cls.RESULTS_DIR = cls.ROOT_DIR / "results"
            cls.PLOTS_DIR = cls.RESULTS_DIR / "plots"
            cls.LOGS_DIR = cls.RESULTS_DIR / "logs"
            cls.DATA_DIR = cls.ROOT_DIR / "data"
            
            # Create directories
            cls._create_directories()
            
            cls.INITIALIZED = True
            logger.info(f"Configuration initialized: {cls.ROOT_DIR}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing configuration: {e}")
            return False
    
    @classmethod
    def _create_directories(cls):
        """
        Create all required directories if they don't exist.
        """
        for directory in [cls.RESULTS_DIR, cls.PLOTS_DIR, cls.LOGS_DIR, cls.DATA_DIR]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directory checked/created: {directory}")
            except Exception as e:
                logger.warning(f"Error creating directory {directory}: {e}")
    
    @classmethod
    def get_plot_path(cls, filename):
        """
        Get the full path for a plot file.
        
        Args:
            filename (str): Name of the plot file
            
        Returns:
            Path: The full path to the plot file
        
        Example:
            >>> plot_path = Config.get_plot_path("my_visualization.png")
        """
        if not cls.INITIALIZED:
            cls.initialize()
        return cls.PLOTS_DIR / filename
    
    @classmethod
    def get_log_path(cls, filename):
        """
        Get the full path for a log file.
        
        Args:
            filename (str): Name of the log file
            
        Returns:
            Path: The full path to the log file
        
        Example:
            >>> log_path = Config.get_log_path("experiment.log")
        """
        if not cls.INITIALIZED:
            cls.initialize()
        return cls.LOGS_DIR / filename
    
    @classmethod
    def get_data_path(cls, filename):
        """
        Get the full path for a data file.
        
        Args:
            filename (str): Name of the data file
            
        Returns:
            Path: The full path to the data file
        
        Example:
            >>> data_path = Config.get_data_path("lattice_data.csv")
        """
        if not cls.INITIALIZED:
            cls.initialize()
        return cls.DATA_DIR / filename
    
    @classmethod
    def setup_logging(cls, log_file=None, level=logging.INFO):
        """
        Set up logging configuration for the framework.
        
        Args:
            log_file (str, optional): Name of the log file. If not provided,
                a default name based on the current date will be used.
            level (int, optional): Logging level. Default is INFO.
                
        Returns:
            logging.Logger: Configured logger
        """
        if not cls.INITIALIZED:
            cls.initialize()
        
        if log_file is None:
            import datetime
            date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_file = f"imhk_{date_str}.log"
        
        log_path = cls.get_log_path(log_file)
        
        # Configure root logger
        root_logger = logging.getLogger("imhk")
        root_logger.setLevel(level)
        
        # Add file handler
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(file_formatter)
        root_logger.addHandler(console_handler)
        
        return root_logger

# Auto-initialize on import
Config.initialize()