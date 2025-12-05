"""initializer.py

Logging initializer module.

This module sets up structured logging for the pipeline using both
standard Python logging and structlog, with optional Elasticsearch
integration. It also defines a context variable to track the current
log phase for contextual logging.

Functions:
    initialize_logs(
        logger_level: int,
    ) -> None
        Configures the root logger with the Elasticsearch handler and
        initializes structlog for structured JSON logging.
"""

import contextvars
import logging

import structlog

from components.const import (
    LOGS_FIELD_PHASE_DEFAULT,
    LOGS_FIELD_PHASE_NAME,
)
from components.logs.handlers.elastic_handler import ElasticHandler
from const import LOGS_LOGGER_NAME

# Contextual variable for logging messages
logs_phase = contextvars.ContextVar(
    LOGS_FIELD_PHASE_NAME,
    default=LOGS_FIELD_PHASE_DEFAULT,
)


def initialize_logs(
    logger_level: int,
) -> None:
    """Initialize logging configuration for the pipeline.

    This function sets up Elasticsearch handler for logs and
    configures structlog for structured logging.

    Args:
        logger_level (int): Logging level.

    Returns:
        None
    """
    # Retrieve logger and configure it
    logger = logging.getLogger(LOGS_LOGGER_NAME)
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
