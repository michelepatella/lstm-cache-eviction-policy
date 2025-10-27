import atexit
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

from components.const import (
    LOGS_ELASTIC_ACTIONS_INDEX_NAME,
    LOGS_ELASTIC_ACTIONS_SOURCE_NAME,
    LOGS_ELASTIC_BULK_SIZE,
    LOGS_ELASTIC_ENDPOINT_ENV_VAR_NAME,
    LOGS_ELASTIC_INDEX_NAME_ENV_VAR_NAME,
    LOGS_ELASTIC_LEVEL_FIELD_NAME,
    LOGS_ELASTIC_LOGGER_FIELD_NAME,
    LOGS_ELASTIC_MESSAGE_FIELD_NAME,
    LOGS_ELASTIC_TIMESTAMP_FIELD_NAME,
    LOGS_ELASTIC_TOKEN_ENV_VAR_NAME,
    LOGS_STANDARD_ATTRS,
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

    Attributes:
         buffer (list): Internal list used to temporarily store log
                        documents before sending them to Elasticsearch.
                        When the number of logs in the buffer reaches
                        the bulk size, all documents are indexed
                        and the buffer is cleared.
         index (str): Elasticsearch index name where logs will be
                      sent to.
    """

    def __init__(self: "ElasticHandler") -> None:
        """Initializes the ElasticHandler instance.

        This function creates an ElasticHandler instance,
        initializing an internal buffer to accumulate logs
        before being sent to Elasticsearch as soon as the
        buffer size reached the predefined (and initialized)
        bulk size. Additionally, the function initializes
        the Elasticsearch index to send logs to. Finally,
        the function sets a listener to ensure that all
        remaining logs in the buffer are sent at
        the exit of the program.

        Args:
            self ("ElasticHandler"): Current class instance.
        """
        super().__init__()
        self.buffer = []
        self.bulk_size = LOGS_ELASTIC_BULK_SIZE
        self.index = os.environ.get(LOGS_ELASTIC_INDEX_NAME_ENV_VAR_NAME)

        # Send all remaining logs in the buffer
        # at the end of the program
        atexit.register(self.flush_buffer)

    def emit(
        self: "ElasticHandler",
        record: logging.LogRecord,
    ) -> None:
        """Emit a log record to Elasticsearch.

        This function prepares each document from the given
        record, appends it to the buffer and sends it to Elasticsearch
        as soon as the buffer reaches predefined bulk size.

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

        # Append document to buffer
        self.buffer.append(doc)

        # Each bulk size requests send
        # all the documents in the buffer
        # to Elasticsearch index
        if len(self.buffer) >= self.bulk_size:
            self.flush_buffer()

    def flush_buffer(
        self: "ElasticHandler",
    ) -> None:
        """Send all buffered log records to Elasticsearch.

        This function takes all log documents currently stored
        in the internal buffer and indexes them into the
        configured Elasticsearch index using the bulk API.
        After successful indexing, the buffer is cleared.

        Args:
            self ("ElasticHandler"): Current class instance.

        Returns:
            None
        """
        # Prepare documents to be sent
        # by reading them from bulk
        actions = [
            {
                LOGS_ELASTIC_ACTIONS_INDEX_NAME: self.index,
                LOGS_ELASTIC_ACTIONS_SOURCE_NAME: d,
            }
            for d in self.buffer
        ]

        # Send documents
        helpers.bulk(es, actions)

        # Clear the buffer
        self.buffer.clear()
