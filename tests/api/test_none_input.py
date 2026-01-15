"""test_none_input.py

This module contains a unit test focused on verifying the robustness of the
API gateway against invalid input types.

It ensures that the API correctly rejects payloads containing null values
for mandatory input fields.

Functions:
    test_api_none_input() -> None:
        Verifies that the API returns a 422 error when receiving None inputs.
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


@pytest.mark.api_none_input
def test_api_none_input() -> None:
    """Tests the API with null (None) input fields.

    This test sends a POST request where mandatory fields are explicitly
    set to None. It verifies that the validation layer correctly
    intercepts the request and returns an Unprocessable Entity status code
    (422), preventing invalid data from entering the inference pipeline.

    Returns:
        None
    """
    # Send API request with None input
    response = requests.post(
        GATEWAY_API_FULL_URL,
        json={
            API_PARAM_KEYS_IN_CACHE_NAME: None,
            API_PARAM_LAST_ACCESSES_NAME: None,
            API_PARAM_USER_API_KWARGS_NAME: None,
        },
    )

    # Assert API failed
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


if __name__ == "__main__":
    test_api_none_input()
