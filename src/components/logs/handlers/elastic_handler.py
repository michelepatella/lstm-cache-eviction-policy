"""elastic_handler.py

Logging handler for Elasticsearch.

This module provides the `ElasticHandler` class, a custom logging
handler that buffers log records and sends them to an Elasticsearch
index. It supports both synchronous and asynchronous flushing
of logs, thread-safe buffering, and preserves extra fields in
the log records.

Classes:
    ElasticHandler(logging.Handler)
        Custom logging handler that accumulates logs and
        sends them to Elasticsearch using the bulk API.
"""

import logging
import os
import threading
from datetime import datetime, timezone

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import BulkIndexError

from components.const import (
    LOGS_ACTIONS_FIELD_INDEX_NAME,
    LOGS_ACTIONS_FIELD_SOURCE_NAME,
    LOGS_ENV_VAR_ELASTIC_ENDPOINT_NAME,
    LOGS_ENV_VAR_ELASTIC_INDEX_NAME,
    LOGS_ENV_VAR_ELASTIC_TOKEN_NAME,
    LOGS_FIELD_LEVEL_NAME,
    LOGS_FIELD_MESSAGE_NAME,
    LOGS_FIELD_STANDARD_NAMES,
    LOGS_FIELD_TIMESTAMP_NAME,
    LOGS_THREAD_DAEMON,
)

# Load environment variables
load_dotenv()

# Configure Elasticsearch
es = Elasticsearch(
    hosts=[os.getenv(LOGS_ENV_VAR_ELASTIC_ENDPOINT_NAME)],
    api_key=os.getenv(LOGS_ENV_VAR_ELASTIC_TOKEN_NAME),
)


class ElasticHandler(logging.Handler):
    """Logging handler that sends logs to Elasticsearch.

    This handler takes standard log records and indexes them into
    an Elasticsearch index.

    Attributes:
         buffer (list): Internal list used to temporarily store log
                        documents before sending them to Elasticsearch.
         index (str): Elasticsearch index name where logs will be
                      sent to.
    """

    def __init__(self: "ElasticHandler") -> None:
        """Initializes the ElasticHandler instance.

        This function creates an ElasticHandler instance,
        initializing an internal buffer to accumulate logs
        before being sent to Elasticsearch. Additionally,
        the function initializes the Elasticsearch index to
        send logs to.

        Args:
            self ("ElasticHandler"): Current class instance.
        """
        super().__init__()
        self.buffer = []
        self.index = os.getenv(LOGS_ENV_VAR_ELASTIC_INDEX_NAME)

    def emit(
        self: "ElasticHandler",
        record: logging.LogRecord,
    ) -> None:
        """Emit a log record to Elasticsearch.

        This function prepares each document from the given
        record, appending it to the buffer to be sent to Elasticsearch
        later.

        Args:
            self ("ElasticHandler"): Current class instance.
            record (logging.LogRecord): The log record to emit.

        Returns:
            None
        """
        # Prepare doc to send to Elasticsearch
        doc = {
            LOGS_FIELD_TIMESTAMP_NAME: datetime.now(
                timezone.utc,
            ).isoformat(),
            LOGS_FIELD_LEVEL_NAME: record.levelname,
            LOGS_FIELD_MESSAGE_NAME: record.getMessage(),
            **{
                k: v
                for k, v in record.__dict__.items()
                if k not in LOGS_FIELD_STANDARD_NAMES
            },
        }

        # Append document to buffer
        self.buffer.append(doc)

    def flush_buffer_sync(
        self: "ElasticHandler",
    ) -> None:
        """Send all buffered log records to Elasticsearch
        synchronously.

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
        # by reading them from buffer
        actions = [
            {
                LOGS_ACTIONS_FIELD_INDEX_NAME: self.index,
                LOGS_ACTIONS_FIELD_SOURCE_NAME: d,
            }
            for d in self.buffer
        ]

        # Send documents
        try:
            helpers.bulk(es, actions)
        except BulkIndexError as e:
            print(f"{len(e.errors)} document(s) failed to index:")
            for error in e.errors:
                print(error)
            raise

        # Clear the buffer
        self.buffer.clear()

    def flush_buffer_async(self: "ElasticHandler") -> None:
        """Send all buffered log records to Elasticsearch
           asynchronously.

        This method launches a separate daemon thread to flush
        all accumulated log records in the internal buffer to
        the configured Elasticsearch index using the bulk API.

        Args:
            self ("ElasticHandler"): Current class instance.

        Returns:
            None
        """
        thread = threading.Thread(
            target=self.flush_buffer_sync,
            daemon=LOGS_THREAD_DAEMON,
        )
        thread.start()
