"""Biblical Counsellor Package

A scripture-based counselling assistant using RAG architecture.
"""

__version__ = "0.1.0"
__author__ = "Lukorito"
__email__ = "jonesdeelder@gmail.com"
__description__ = "Biblical Counsellor - AI assistant providing scripture-based guidance"
__license__ = "MIT"
# __all__ = ['config', 'log_manager'] # Generally not necessary for a top-level __init__.py unless explicitly exposing for 'from package import *'

# Package initialization
import os
import logging
from importlib.metadata import version

# 1. Import config first, as it's needed by log_manager.configure_logging
from config.settings import config

# 2. Import log_manager. The log_manager object is now available.
from utils.log_manager import log_manager

# Configure the package's logger immediately after imports
# This ensures that any subsequent logging messages from this __init__.py
# and other package modules use your custom configuration.
# We pass config.logging_level directly.
app_logger = log_manager.configure_logging(
    name="biblical_counselor",  # Use the main application logger name
    console_level=config.logging_level,
    file_level=config.logging_level
)

try:
    __version__ = version("biblical-counselor")
except ImportError:
    # If not installed as a package, use the local version.
    # We should log this with our configured logger.
    app_logger.debug(
        "Package 'biblical-counselor' not installed, using local __version__.")
    pass

# Now you can use app_logger for package-level messages
app_logger.info(f"Initialising Biblical Counsellor v{__version__}")


def setup_package():
    """Initialise core package functionality."""
    # Ensure required directories exist.
    # config.data_path should be created.
    # config.logging_file's directory is already handled by log_manager.configure_logging.
    os.makedirs(config.data_path, exist_ok=True)

    app_logger.debug("Package initialisation complete.")


setup_package()
