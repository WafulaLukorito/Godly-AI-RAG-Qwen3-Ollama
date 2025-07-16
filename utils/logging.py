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

    def configure_logging(self,
                          name: str = "biblical_counselor",
                          console_level: int = logging.INFO,
                          file_level: int = logging.DEBUG) -> logging.Logger:
        """
        Configure dual logging (console + file) with timestamped filenames.

        Args:
            name: Logger name prefix
            console_level: Console logging level
            file_level: File logging level

        Returns:
            Configured logger instance
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{self.log_dir}/{name}_{timestamp}.log"

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Capture all levels

        # Clear existing handlers
        logger.handlers.clear()

        # File handler (detailed)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        # Console handler (simpler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Add exception hook
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


# Singleton instance for easy access
log_manager = LogManager()
