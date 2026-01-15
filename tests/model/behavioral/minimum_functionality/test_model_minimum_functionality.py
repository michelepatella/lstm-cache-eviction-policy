"""test_model_minimum_functionality.py

Module dedicated to executing minimum functionality tests against the trained model.

This module orchestrates the minimum functionality tests, utilizing predefined sequences
(features and corresponding keys) and verifying that the model's prediction
matches the expected next key in the sequence.

Functions:
    test_model_minimum_functionality() -> None
        Executes the minimum functionality tests.
"""

import pytest

from tests.model.behavioral.minimum_functionality.helpers import (
    model_minimum_functionality_on_sequence,
)
from tests.model.helpers import initialize_inference_environment


@pytest.mark.model_behavioral_minimum_functionality
@pytest.mark.after_model_training
def test_model_minimum_functionality() -> None:
    """Performs minimum functionality tests.

    The test initializes the inference environment and then checks the model's
    predictions against known sequences of features and keys, in order to verify
    that the model is able to provide the minimum, expected functionalities.

    Returns:
        None
    """
    # ----------------------------
    # Setup
    # ----------------------------
    # Initialize the inference environment
    (testing_loader, model, device, pipeline_config, tests_config) = (
        initialize_inference_environment()
    )

    # ----------------------------
    # Minimum Functionality Tests
    # ----------------------------
    # Example of minimum functionality test: test if the
    # model is learnt a simple sequential pattern, by defining
    # both features and keys sequences and the expected key
    features_sequence = [
        [0.6223982138696689, -0.7827007495664265, 0.04, 0.0],
        [0.6124288405323478, -0.7905257208239362, 0.04, 0.0],
        [0.6104719796770406, -0.7920378539117906, 0.04, 0.0],
        [0.6097757808312311, -0.7925739694890708, 0.04, 0.0],
        [0.6066941192900404, -0.794935371976164, 0.04, 0.0],
        [0.6006270322950682, -0.7995293415981799, 0.04, 0.0],
        [0.6002082961660248, -0.7998437354968015, 0.04, 0.0],
        [0.5998344491575055, -0.8001241363712958, 0.04, 0.0],
        [0.5938129235087608, -0.8046031393637354, 0.04, 0.0],
        [0.592508177368534, -0.8055644355055763, 0.04, 0.0],
        [0.591132408564136, -0.8065745319219815, 0.04, 0.0],
        [0.5902152470971341, -0.8072459117852929, 0.04, 0.0],
        [0.5897141331658644, -0.8076120610444306, 0.04, 0.0],
        [0.5868317927790053, -0.809708865570693, 0.04, 0.0],
        [0.5788425550794389, -0.8154393272519462, 0.04, 0.0],
        [0.5700334592950794, -0.821621479322495, 0.04, 0.0],
        [0.5631242943025839, -0.8263722098220734, 0.04, 0.0],
        [0.5578902298076331, -0.8299147495286406, 0.04, 0.0],
        [0.5509957413432837, -0.8345080544977174, 0.04, 0.0],
        [0.5502643809076128, -0.834990485637029, 0.04, 0.0],
        [0.5491437378353997, -0.8357279193590255, 0.04, 0.0],
        [0.5490740116917225, -0.8357737311526117, 0.04, 0.0],
        [0.5412869715427089, -0.8408379240008877, 0.04, 0.0],
        [0.5380683934051343, -0.8429011828313077, 0.04, 0.0],
        [0.5360692407649906, -0.8441740158910641, 0.04, 0.0],
    ]
    keys_sequence = [
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
    ]
    expected_key = 70

    # Test if the model predicts the expected class
    # over a simple input
    model_minimum_functionality_on_sequence(
        (testing_loader, model, device, pipeline_config, tests_config),
        features_sequence,
        keys_sequence,
        expected_key,
    )


if __name__ == "__main__":
    test_model_minimum_functionality()
