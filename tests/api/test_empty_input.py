"""test_empty_input.py

This module contains an integration test for the API.

It focuses on verifying the behavior of the API when subjected
to empty payloads, to ensure it handles errors and returns the
expected HTTP status code.

Functions:
    test_api_empty_input() -> None:
        Tests the API response when the input is empty.
"""

from http import HTTPStatus

import pytest
import requests

from components.const import (
    API_PARAM_KEYS_IN_CACHE_NAME,
    API_PARAM_LAST_ACCESSES_NAME,
    API_PARAM_USER_API_KWARGS_NAME,
)
from const import GATEWAY_API_FULL_URL


@pytest.mark.api_empty_input
def test_api_empty_input() -> None:
    """Tests the API with empty input fields.

    This test verifies that the API correctly identifies and handles a
    scenario where no keys are in the cache and no last accesses are
    provided. It asserts that the system returns an Internal Server Error
    status within the response message, as the pipeline cannot proceed
    without data.

    Returns:
        None
    """
    # Send API request with empty input
    response = requests.post(
        GATEWAY_API_FULL_URL,
        json={
            API_PARAM_KEYS_IN_CACHE_NAME: [],
            API_PARAM_LAST_ACCESSES_NAME: [],
            API_PARAM_USER_API_KWARGS_NAME: {},
        },
    )

    # Assert API's message status code
    assert str(HTTPStatus.INTERNAL_SERVER_ERROR) in response.json().get(
        "message",
    )


if __name__ == "__main__":
    test_api_empty_input()
