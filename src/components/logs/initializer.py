import contextvars
import logging
import os

import structlog
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from components.const import (
    LOGS_DEBUG_FILE_PATH,
    LOGS_DEFAULT_PHASE,
    LOGS_ELASTIC_ENDPOINT_ENV_VAR_NAME,
    LOGS_ELASTIC_TOKEN_ENV_VAR_NAME,
    LOGS_ERROR_FILE_PATH,
    LOGS_FILE_BACKUP_COUNT,
    LOGS_FILE_BASE_LEVEL,
    LOGS_FILE_MAX_BYTES,
    LOGS_FORMAT,
    LOGS_INFO_FILE_PATH,
    LOGS_LOGGER_NAME,
    LOGS_PHASE_NAME,
)
from components.logs.handlers.elastic_handler import ElasticHandler
from components.logs.handlers.file_handler_builder import (
    build_logs_file_handler,
)

# Load environment variables
load_dotenv()

# Contextual variable for logging messages
logs_phase = contextvars.ContextVar(
    LOGS_PHASE_NAME,
    default=LOGS_DEFAULT_PHASE,
)

# Configure Elasticsearch
es = Elasticsearch(
    hosts=[os.environ.get(LOGS_ELASTIC_ENDPOINT_ENV_VAR_NAME)],
    api_key=os.environ.get(LOGS_ELASTIC_TOKEN_ENV_VAR_NAME),
)


def initialize_logs(
    debug_path=LOGS_DEBUG_FILE_PATH,
    info_path=LOGS_INFO_FILE_PATH,
    error_path=LOGS_ERROR_FILE_PATH,
    base_level=LOGS_FILE_BASE_LEVEL,
    max_bytes=LOGS_FILE_MAX_BYTES,
    backup_count=LOGS_FILE_BACKUP_COUNT,
    logs_format=LOGS_FORMAT,
    logger_name: str = LOGS_LOGGER_NAME,
):
    """Initialize logging configuration for the pipeline.

    This function sets up file handlers for debug, info, and error levels,
    attaches an ElasticSearch handler, and configures structlog for
    structured logging. It ensures that logs are saved to files, sent to
    Elasticsearch, and formatted in JSON.

    Args:
        debug_path (str): Path for the debug-level log file.
        info_path (str): Path for the info-level log file.
        error_path (str): Path for the error-level log file.
        base_level (int): Base logging level for the logger.
        max_bytes (int): Maximum size in bytes for each log file
                         before rotation.
        backup_count (int): Number of backup files to keep during
                            rotation.
        logs_format (str): Format string for log messages.
        logger_name (str): Name of the logger to configure.

    Returns:
        None
    """
    # For each logging level, build
    # its own file handler
    debug_file_handler = build_logs_file_handler(
        debug_path,
        logging.DEBUG,
        max_bytes,
        backup_count,
        logs_format,
    )
    info_file_handler = build_logs_file_handler(
        info_path,
        logging.INFO,
        max_bytes,
        backup_count,
        logs_format,
    )
    error_file_handler = build_logs_file_handler(
        error_path,
        logging.ERROR,
        max_bytes,
        backup_count,
        logs_format,
    )

    # Retrieve logger and configure it
    logger = logging.getLogger(logger_name)
    logger.setLevel(base_level)

    # Add handlers to logger
    logger.addHandler(debug_file_handler)
    logger.addHandler(info_file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(ElasticHandler())

    # To ensure structured logs
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )
