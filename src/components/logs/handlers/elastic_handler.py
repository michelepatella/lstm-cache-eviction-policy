import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from components.const import (
    LOGS_ELASTIC_INDEX_NAME_ENV_VAR_NAME,
    LOGS_ELASTIC_LEVEL_FIELD_NAME,
    LOGS_ELASTIC_LOGGER_FIELD_NAME,
    LOGS_ELASTIC_MESSAGE_FIELD_NAME,
    LOGS_ELASTIC_TIMESTAMP_FIELD_NAME,
    LOGS_STANDARD_ATTRS, LOGS_ELASTIC_ENDPOINT_ENV_VAR_NAME, LOGS_ELASTIC_TOKEN_ENV_VAR_NAME,
)

# Load environment variables
load_dotenv()

# Configure Elasticsearch
es = Elasticsearch(
    hosts=[os.environ.get(LOGS_ELASTIC_ENDPOINT_ENV_VAR_NAME)],
    api_key=os.environ.get(LOGS_ELASTIC_TOKEN_ENV_VAR_NAME),
)

class ElasticHandler(logging.Handler):
    """Logging handler that sends logs to Elasticsearch.

    This handler takes standard log records and indexes them into
    an Elasticsearch index.
    """

    def emit(self: "ElasticHandler", record: logging.LogRecord) -> None:
        """Emit a log record to Elasticsearch.

        This function emits a given log record to an Elasticsearch index.

        Args:
            self ("ElasticHandler"): Current class instance.
            record (logging.LogRecord): The log record to emit.

        Returns:
            None
        """
        # Prepare doc to send to Elasticsearch
        doc = {
            LOGS_ELASTIC_TIMESTAMP_FIELD_NAME: datetime.now(
                timezone.utc,
            ).isoformat(),
            LOGS_ELASTIC_LEVEL_FIELD_NAME: record.levelname,
            LOGS_ELASTIC_LOGGER_FIELD_NAME: record.name,
            LOGS_ELASTIC_MESSAGE_FIELD_NAME: record.getMessage(),
            **{
                k: v
                for k, v in record.__dict__.items()
                if k not in LOGS_STANDARD_ATTRS
            },
        }

        # Send doc to Elasticsearch at predefined index
        es.index(
            index=os.environ.get(LOGS_ELASTIC_INDEX_NAME_ENV_VAR_NAME),
            document=doc,
        )
