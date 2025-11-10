"""grafana_loki_handler.py

Logging handler for Grafana Loki.

This module provides the `GrafanaLokiHandler` class, a custom logging
handler that buffers log records and sends them to a Grafana Loki
HTTP endpoint. It supports both synchronous and asynchronous flushing
of logs, thread-safe buffering, and correctly formats log records
for the Grafana Loki Push API.

Classes:
    GrafanaLokiHandler(logging.Handler):
        Custom logging handler that accumulates logs and
        sends them to Grafana Loki using the v1/push API.
"""

import json
import logging
import threading

import requests

from components.const import (
    GRAFANA_LOKI_TOKEN,
    LOGS_GRAFANA_LOKI_URL,
    LOGS_GRAFANA_LOKI_USER_ID,
    LOGS_GRAFANA_LOKI_FIELD_LEVEL_NAME,
    LOGS_GRAFANA_LOKI_FIELD_MESSAGE_NAME,
    LOGS_GRAFANA_LOKI_FIELD_PHASE_NAME,
    LOGS_FIELD_STANDARD_NAMES,
    LOGS_GRAFANA_LOKI_API_HEADERS,
    LOGS_GRAFANA_LOKI_API_PAYLOAD_STREAM_NAME,
    LOGS_GRAFANA_LOKI_API_PAYLOAD_STREAMS_NAME,
    LOGS_GRAFANA_LOKI_API_PAYLOAD_VALUES_NAME,
    LOGS_GRAFANA_LOKI_ASYNC_THREAD_DAEMON,
    TIME_NANOSECONDS_IN_SECOND,
)
from components.logs.initializer import logs_phase

# To ensure safety in multi-thread
# environments
_buffer_lock = threading.Lock()


class GrafanaLokiHandler(logging.Handler):
    """Logging handler that sends logs to Grafana Loki.

    This handler takes standard log records and formats them
    according to the Grafana Loki Push API (v1/push).

    Attributes:
         buffer (list[tuple[str, str]]): Internal list used to temporarily store
                                         log documents (tuples of [timestamp, line])
                                         before sending them to Grafana Loki.
         labels (dict): Default stream labels to be included in every
                        log stream sent to Grafana Loki.
    """

    def __init__(
        self: "GrafanaLokiHandler",
        labels: dict[str, str] = None,
    ) -> None:
        """Initializes the GrafanaLokiHandler instance.

        Args:
            self ("GrafanaLokiHandler"): Current class instance.
            labels (dict[str, str]): Initial stream labels.
        """
        super().__init__()
        self.buffer: list[tuple[str, str]] = []
        self.labels = (
            labels
            if labels is not None
            else {LOGS_GRAFANA_LOKI_FIELD_PHASE_NAME: logs_phase.get()}
        )
        self._url = LOGS_GRAFANA_LOKI_URL
        self._auth = (LOGS_GRAFANA_LOKI_USER_ID, GRAFANA_LOKI_TOKEN)

    def emit(
        self: "GrafanaLokiHandler",
        record: logging.LogRecord,
    ) -> None:
        """Emit a log record to the internal buffer.

        This function formats the log line and the timestamp,
        appending it to the thread-safe buffer.

        Args:
            self ("GrafanaLokiHandler"): Current class instance.
            record (logging.LogRecord): The log record to emit.

        Returns:
            None
        """
        # Prepare the log data
        log_data = {
            LOGS_GRAFANA_LOKI_FIELD_LEVEL_NAME: record.levelname,
            LOGS_GRAFANA_LOKI_FIELD_MESSAGE_NAME: record.getMessage(),
            **{
                k: v
                for k, v in record.__dict__.items()
                if k not in LOGS_FIELD_STANDARD_NAMES
            },
        }

        # Prepare the log to be sent to Grafana Loki
        # as a couple (log_line, timestamp_ns), where
        # the first is the log data as a string
        log_line = json.dumps(log_data)
        timestamp_ns = str(int(record.created * TIME_NANOSECONDS_IN_SECOND))
        log_record_tuple = (timestamp_ns, log_line)

        # Add the just created log to
        # the buffer
        self.buffer.append(log_record_tuple)

    def flush_buffer_sync(
        self: "GrafanaLokiHandler",
    ) -> None:
        """Send all buffered log records to Grafana Loki synchronously.

        This function formats the buffer into the Grafana Loki Push API
        format and sends it via HTTP POST. After successful indexing,
        the buffer is cleared.

        Args:
            self ("GrafanaLokiHandler"): Current class instance.

        Returns:
            None
        """
        # Build payload for Grafana Loki API
        payload = {
            LOGS_GRAFANA_LOKI_API_PAYLOAD_STREAMS_NAME: [
                {
                    LOGS_GRAFANA_LOKI_API_PAYLOAD_STREAM_NAME: self.labels,
                    LOGS_GRAFANA_LOKI_API_PAYLOAD_VALUES_NAME: self.buffer,
                },
            ],
        }

        # Send payload to Grafana Loki
        response = requests.post(
            url=self._url,
            auth=self._auth,
            json=payload,
            headers=LOGS_GRAFANA_LOKI_API_HEADERS,
        )
        response.raise_for_status()

        # Clear the buffer
        self.buffer.clear()

    def flush_buffer_async(self: "GrafanaLokiHandler") -> None:
        """Send all buffered log records to Grafana Loki asynchronously.

        This method launches a separate daemon thread to flush
        all accumulated log records in the internal buffer to
        the configured Grafana Loki endpoint.

        Args:
            self ("GrafanaLokiHandler"): Current class instance.

        Returns:
            None
        """
        # Create and run the thread
        thread = threading.Thread(
            target=self.flush_buffer_sync,
            daemon=LOGS_GRAFANA_LOKI_ASYNC_THREAD_DAEMON,
        )
        thread.start()
