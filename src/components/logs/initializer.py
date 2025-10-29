import contextvars
import logging

import structlog

from components.const import (
    LOGS_DEFAULT_LEVEL,
    LOGS_DEFAULT_PHASE,
    LOGS_LOGGER_NAME,
    LOGS_PHASE_NAME,
)
from components.logs.handlers.elastic_handler import ElasticHandler

# Contextual variable for logging messages
logs_phase = contextvars.ContextVar(
    LOGS_PHASE_NAME,
    default=LOGS_DEFAULT_PHASE,
)


def initialize_logs(
    logger_name: str = LOGS_LOGGER_NAME,
    logger_level: int = LOGS_DEFAULT_LEVEL,
) -> None:
    """Initialize logging configuration for the pipeline.

    This function sets up Elasticsearch handler for logs and
    configures structlog for structured logging.

    Args:
        logger_name (str): Name of the logger to configure.
        logger_level (int): Logging level.

    Returns:
        None
    """
    # Retrieve logger and configure it
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)

    # Add handlers to logger
    logger.addHandler(ElasticHandler())

    # To ensure structured logs
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )
