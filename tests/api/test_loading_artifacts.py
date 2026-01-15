"""test_loading_artifacts.py

This module contains a unit test focused on the validation and integrity
of the artifacts used by the API.

It ensures that artifacts are correctly loaded into memory, initialized, and
conform to the expected architectural specifications.

Functions:
    test_api_loading_artifacts() -> None:
        Verifies that the predictive model is correctly loaded and typed.
"""

import pytest

from api.services.predictor.predictor_service import model
from components.model.lstm import LSTM


@pytest.mark.api_loading_artifacts
def test_api_loading_artifacts() -> None:
    """Verifies the correct loading and typing of the model artifact.

    This test ensures that the model instance in the predictor
    service is not None and has been correctly instantiated as an
    LSTM object.

    Returns:
        None
    """
    # Assert the model is not None
    assert model is not None

    # Assert the model is of expected type
    assert isinstance(model, LSTM)


if __name__ == "__main__":
    test_api_loading_artifacts()
