"""Biblical Counselor Package

A scripture-based counseling assistant using RAG architecture.
"""

__version__ = "0.1.0"
__author__ = "Lukorito"
__email__ = "jonesdeelder@gmail.com"
__description__ = "Biblical Counselor - AI assistant providing scripture-based guidance"
__license__ = "MIT"
__all__ = ['config', 'log_manager']  # Public API

# Package initialization
import os
import logging
from importlib.metadata import version

try:
    __version__ = version("biblical-counselor")
except ImportError:
    pass  # using local version if not installed

# Initialize package components
from config.settings import config  # Correct import path
from utils.logging import log_manager


def setup_package():
    """Initialize core package functionality."""
    # Configure package-level logging
    logger = logging.getLogger(__name__)

    # Ensure required directories exist
    os.makedirs(config.data_path, exist_ok=True)
    os.makedirs(os.path.dirname(config.logging_file), exist_ok=True)

    logger.info(f"Initializing Biblical Counselor v{__version__}")
    logger.debug("Package initialization complete")


# Run setup when package is imported
setup_package()
