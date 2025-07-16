import logging
from datetime import datetime
import os
import sys
from typing import Optional, Dict, Any

# No module-level import of config to avoid circular dependency
# from config.settings import config # DO NOT DO THIS HERE


class LogManager:
    """Centralized logging management with file and console output."""

    def __init__(self):
        pass

    def configure_logging(
        self,
        name: str = "biblical_counselor",
        console_level: str = None,
        file_level: str = None
    ) -> logging.Logger:
        """
        Configure dual logging (console + file) with rotation
        and coloured output.

        Args:
            name: Logger name (default: "biblical_counselor")
            console_level: Console logging level
            (default: config.logging_level)
            file_level: File logging level (default: config.logging_level)

        Returns:
            Configured logger instance
        """
        # Import settings here to avoid circular imports at module load time
        # This ensures 'config' is available when configure_logging is called
        from config.settings import config

        # Set defaults from config if not provided
        console_level = console_level or config.logging_level
        file_level = file_level or config.logging_level

        # Convert string levels to logging constants
        try:
            console_level = getattr(logging, console_level.upper())
            file_level = getattr(logging, file_level.upper())
        except AttributeError:
            print(
                f"Warning: Invalid logging level specified. Defaulting to INFO. Console: {console_level}, File: {file_level}")
            console_level = logging.INFO
            file_level = logging.INFO

        logger = logging.getLogger(name)
        # Base level captures all messages for handlers to filter
        logger.setLevel(logging.DEBUG)

        # Clear existing handlers to avoid duplication if called multiple times
        if logger.handlers:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

        # Ensure log directory for the specific file exists
        # This logic is now correctly placed where 'config' is available
        log_file_dir = os.path.dirname(config.logging_file)
        os.makedirs(log_file_dir, exist_ok=True)

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
        # file_handler.addFilter(UTF8EncodeFilter()) # Only if specific encoding issues persist

        # Coloured console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(ColouredFormatter(
            fmt='%(asctime)s - %(levelname)8s - %(message)s',
            datefmt='%H:%M:%S'
        ))

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        sys.excepthook = self._handle_uncaught_exception

        return logger

    def _handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """Global exception handler."""
        logger = logging.getLogger("biblical_counselor")
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback))


class ColouredFormatter(logging.Formatter):
    """Coloured log formatter for console output."""

    COLOURS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[1;31m',  # Bold Red
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record):
        """Apply colour coding based on log level."""
        level_colour = self.COLOURS.get(record.levelname, self.COLOURS['INFO'])
        message = super().format(record)
        return f"{level_colour}{message}{self.COLOURS['RESET']}"


class UTF8EncodeFilter(logging.Filter):
    """Ensure log messages are UTF-8 encoded."""

    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = record.msg.encode('utf-8', 'replace').decode('utf-8')
        return True


# Singleton instance for easy access
log_manager = LogManager()

# The initialisation of the logger itself (calling configure_logging)
# should happen in your main application entry point, after 'config' is ready.
