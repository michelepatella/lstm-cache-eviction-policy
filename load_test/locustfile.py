"""locustfile.py

This module defines the load testing logic using Locust to evaluate the
performance and robustness of the API.

It simulates concurrent users performing various tasks, including health
checks and both valid and invalid prediction requests. The script leverages
the actual processed dataset to generate realistic input sequences and
randomly corrupts data to test the API's validation and error-handling
capabilities.

Classes:
    DataEvictionUser(HttpUser):
        Simulates a client that interacts with the API.
"""

from helpers import get_random_api_input
from locust import HttpUser, task

from components.const import (
    API_PARAM_KEYS_IN_CACHE_NAME,
    API_PARAM_LAST_ACCESSES_NAME,
    API_PARAM_USER_API_KWARGS_NAME,
)
from components.dataset.io.loader import load_dataset
from components.dataset.io.locator import get_dataset_abs_path
from const import GATEWAY_API_FULL_URL
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import DATASET_PROCESSED_TYPE

# Prepare configuration
pipeline_config = prepare_pipeline_config()
data_mode = pipeline_config.data.general.mode
seq_len = pipeline_config.model.sequence.length

# Load processed dataset
dataset_abs_path = get_dataset_abs_path(DATASET_PROCESSED_TYPE, data_mode)
dataset = load_dataset(dataset_abs_path)


class DataEvictionUser(HttpUser):
    """Simulate a user interacting with the API.

    This class defines the behavior of virtual users during a load test,
    distributing requests between health checks, standard prediction
    scenarios, and error-inducing edge cases.
    """

    def _call_api(
        self: "DataEvictionUser",
        last_accesses: list[tuple[float, int]],
        keys_in_cache: list[int],
    ) -> None:
        """Executes a request to the API.

        This method encapsulates the logic for sending
        the request payload to the API endpoint.

        Args:
            self (DataEvictionUser): The instance of the Locust user.
            last_accesses (list[tuple[float, int]]): The sequence of recent
                                                     accesses.
            keys_in_cache (list[int]): The list of keys currently stored
                                       in the cache.

        Returns:
            None
        """
        self.client.post(
            GATEWAY_API_FULL_URL,
            json={
                API_PARAM_KEYS_IN_CACHE_NAME: keys_in_cache,
                API_PARAM_LAST_ACCESSES_NAME: last_accesses,
                API_PARAM_USER_API_KWARGS_NAME: {},
            },
        )

    # Health check: 10% of traffic
    @task(1)
    def healthcheck(self: "DataEvictionUser") -> None:
        """Perform a basic health check on the API.

        This task verify the availability of the service with a simple
        GET request.

        Args:
            self (DataEvictionUser): The instance of the Locust user.

        Returns:
            None
        """
        self.client.get("/")

    # Valid predictions: 70% of traffic
    @task(7)
    def prediction_valid(self: "DataEvictionUser") -> None:
        """Send a valid request to the API.

        This task fetches access sequences from the dataset and
        submits them to the endpoint to simulate normal production traffic.

        Args:
            self (DataEvictionUser): The instance of the Locust user.

        Returns:
            None
        """
        last_accesses, keys_in_cache = get_random_api_input(dataset, seq_len)
        self._call_api(last_accesses, keys_in_cache)

    # Invalid predictions: 20% of traffic
    @task(2)
    def prediction_invalid(self: "DataEvictionUser") -> None:
        """Send a corrupted request to the API to test error handling.

        This task intentionally introduces NaN values into a randomly
        selected column of the input data, verifying how the API manages
        invalid payloads and validation failures.

        Args:
            self (DataEvictionUser): The instance of the Locust user.

        Returns:
            None
        """
        last_accesses, keys_in_cache = get_random_api_input(
            dataset,
            seq_len,
            corrupt=True,
        )
        self._call_api(last_accesses, keys_in_cache)
