"""test_wrong_input_type.py

This module contains a unit test designed to verify the API's
data validation layer, specifically focusing on type enforcement.

It ensures that the API correctly identifies and rejects payloads containing
incorrect data types for mandatory fields.

Functions:
    test_api_wrong_input_type() -> None:
        Verifies that the API returns a 422 error when receiving invalid data types.
"""

from http import HTTPStatus

import requests

from components.const import (
    API_PARAM_KEYS_IN_CACHE_NAME,
    API_PARAM_LAST_ACCESSES_NAME,
    API_PARAM_USER_API_KWARGS_NAME,
)
from const import GATEWAY_API_FULL_URL


def test_api_wrong_input_type() -> None:
    """Tests the API with invalid data types in the payload.

    This test sends a POST request where the mandatory lists contain
    strings instead of the expected integers and floats. It verifies
    that the API's validation layer triggers an Unprocessable
    Entity (422) status code, ensuring that malformed data does
    not reach the inference services.

    Returns:
        None
    """
    # Send API request with intentionally wrong data types
    response = requests.post(
        GATEWAY_API_FULL_URL,
        json={
            API_PARAM_KEYS_IN_CACHE_NAME: ["a", "b", "c"],
            API_PARAM_LAST_ACCESSES_NAME: [["x", "y"], ["z", "w"]],
            API_PARAM_USER_API_KWARGS_NAME: {},
        },
    )

    # Assert API failed
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
