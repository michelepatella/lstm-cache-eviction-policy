from helpers import (  # noqa
    model_directional_tests_setup,
    test_model_directional_local_feature_perturbations,
)

from tests.const import DATASET_PROCESSED_LOCAL_FEATURE_COLUMNS
from tests.model.helpers import initialize_inference_environment


def test_model_directional() -> None:
    # ----------------------------
    # Setup
    # ----------------------------
    # Initialize the inference environment
    (testing_loader, model, device, pipeline_config, tests_config) = (
        initialize_inference_environment()
    )

    # ----------------------------
    # Directional Tests
    # ----------------------------
    # Test model directional against local feature perturbations,
    # including local frequency and recency (whether predictions
    # over increased/decreased features change accordingly)
    for local_feature in DATASET_PROCESSED_LOCAL_FEATURE_COLUMNS:
        test_model_directional_local_feature_perturbations(
            (testing_loader, model, device, pipeline_config, tests_config),
            local_feature,
        )
