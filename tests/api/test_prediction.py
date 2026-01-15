"""test_prediction.py

This module contains a unit test focused on verifying the end-to-end
API prediction flow.

It verifies that valid inputs produce consistent eviction results according
to the configured pipeline parameters.

Functions:
    test_api_prediction() -> None:
        Validates the API prediction logic using a random sample from the dataset.
"""

from http import HTTPMethod, HTTPStatus

import pytest
import requests

from components.const import (
    API_PARAM_KEYS_IN_CACHE_NAME,
    API_PARAM_LAST_ACCESSES_NAME,
    API_PARAM_USER_API_KWARGS_NAME,
)
from components.dataset.io.loader import load_dataset
from components.dataset.io.locator import get_dataset_abs_path
from const import (
    API_RESPONSE_FIELD_DATA_KEYS_TO_EVICT_NAME,
    API_RESPONSE_FIELD_DATA_NAME,
    GATEWAY_API_FULL_URL,
)
from load_test.helpers import get_random_api_input
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import DATASET_PROCESSED_TYPE


@pytest.mark.api_prediction
def test_api_prediction() -> None:
    """Validates the API prediction pipeline with a realistic input scenario.

    This test sends a POST request to the API with valid parameters, asserting
    that the response is successful (200 OK), the method is POST, and the
    returned keys to evict match the expected count and data type (int).

    Returns:
        None
    """
    # Prepare pipeline configuration
    pipeline_config = prepare_pipeline_config()
    data_mode = pipeline_config.data.general.mode
    seq_len = pipeline_config.model.sequence.length
    api_kwargs = pipeline_config.simulations.api_kwargs

    # Load processed dataset
    dataset_abs_path = get_dataset_abs_path(DATASET_PROCESSED_TYPE, data_mode)
    dataset = load_dataset(dataset_abs_path)

    # Select a valid API input randomly
    last_accesses, keys_in_cache = get_random_api_input(dataset, seq_len)

    # Send a valid API request
    response = requests.post(
        GATEWAY_API_FULL_URL,
        json={
            API_PARAM_KEYS_IN_CACHE_NAME: keys_in_cache,
            API_PARAM_LAST_ACCESSES_NAME: last_accesses,
            API_PARAM_USER_API_KWARGS_NAME: api_kwargs.__dict__,
        },
    )

    # Assert API response's status code, method, length, and type
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == HTTPMethod.POST

    # Extract results
    eviction_data = response.json()[API_RESPONSE_FIELD_DATA_NAME][
        API_RESPONSE_FIELD_DATA_KEYS_TO_EVICT_NAME
    ]

    assert len(eviction_data) == api_kwargs.num_evictions
    for output in eviction_data:
        assert isinstance(output, int)


if __name__ == "__main__":
    test_api_prediction()
