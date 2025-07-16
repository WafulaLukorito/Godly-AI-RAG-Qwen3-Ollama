import logging
from datetime import datetime
import os
import sys
from typing import Optional, Dict, Any
from config import settings as config


class LogManager:
    """Centralized logging management with file and console output."""

    def __init__(self):
        self.log_dir = "logs"
        self._setup_log_directory()

    def _setup_log_directory(self):
        """Ensure logs directory exists."""
        os.makedirs(self.log_dir, exist_ok=True)

    def configure_logging(
        self,
        name: str = "biblical_counselor",
        console_level: str = None,
        file_level: str = None
    ) -> logging.Logger:
    """
    Configure dual logging (console + file) with rotation and colored output.
    
    Args:
        name: Logger name (default: "biblical_counselor")
        console_level: Console logging level (default: config.logging_level)
        file_level: File logging level (default: config.logging_level)
    
    Returns:
        Configured logger instance
    """
    # Import settings directly to avoid circular imports
    from config.settings import config

    # Set defaults from config if not provided
    console_level = console_level or config.logging_level
    file_level = file_level or config.logging_level

    # Convert string levels to logging constants
    console_level = getattr(logging, console_level.upper())
    file_level = getattr(logging, file_level.upper())

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Base level captures all

    # Clear existing handlers to avoid duplication
    logger.handlers.clear()

    # Ensure log directory exists
    log_dir = os.path.dirname(config.logging_file)
    os.makedirs(log_dir, exist_ok=True)

    # File handler with rotation
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        filename=config.logging_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(
        fmt=config.logging_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Colored console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColoredFormatter(
        fmt='%(asctime)s - %(levelname)8s - %(message)s',
        datefmt='%H:%M:%S'
    ))

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Configure root logger for third-party packages
    logging.basicConfig(level=logging.WARNING)

    # Exception handling
    sys.excepthook = self._handle_uncaught_exception

    return logger

    def _handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """Global exception handler."""
        logger = logging.getLogger("biblical_counselor.root")
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback))

        # For production: Could add error reporting service call here


class ColoredFormatter(logging.Formatter):
    """Color-coded log formatter for console output."""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[1;31m',  # Bold Red
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record):
        """Apply color coding based on log level."""
        level_color = self.COLORS.get(record.levelname, self.COLORS['INFO'])
        message = super().format(record)
        return f"{level_color}{message}{self.COLORS['RESET']}"


class UTF8EncodeFilter(logging.Filter):
    """Ensure log messages are UTF-8 encoded."""

    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = record.msg.encode('utf-8', 'replace').decode('utf-8')
        return True


# Singleton instance for easy access
log_manager = LogManager()
