import logging
from logging.handlers import RotatingFileHandler

from pipeline.const import (
    LOGS_FILE_BACKUP_COUNT,
    LOGS_FILE_MAX_BYTES,
    LOGS_FORMAT,
)


def create_logs_file_handler(
    path: str,
    level: int,
    max_bytes: int = LOGS_FILE_MAX_BYTES,
    backup_count: int = LOGS_FILE_BACKUP_COUNT,
    file_format: str = LOGS_FORMAT,
) -> RotatingFileHandler:
    """
    Set up a RotatingFileHandler for logging to a file.

    This function creates and configures a rotating file handler
    with the standard formatter and specified logs level. It is
    used to write logs to disk while limiting file size and
    keeping backups.

    Args:
        path (str): Path to the logs file.
        level (int): Logging level.
        max_bytes (int): Maximum file size in bytes before rotation.
        backup_count (int): Number of backup files to keep.
        file_format (str): Format string for logs messages.

    Returns:
        RotatingFileHandler: Rotating file handler configured.
    """
    # Create a rotating file handler
    file_handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )

    # Set logs level
    file_handler.setLevel(level)

    # Apply formatter
    file_handler.setFormatter(logging.Formatter(file_format))

    return file_handler
