from helpers import (  # noqa
    model_directional_tests_setup,
    test_model_directional_local_feature_perturbations,
)

from components.const import (
    DATASET_COLUMN_LOCAL_FREQUENCY_NAME,
    DATASET_COLUMN_LOCAL_RECENCY_NAME,
)
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
    test_model_directional_local_feature_perturbations(
        (testing_loader, model, device, pipeline_config, tests_config),
        DATASET_COLUMN_LOCAL_FREQUENCY_NAME,
    )
    test_model_directional_local_feature_perturbations(
        (testing_loader, model, device, pipeline_config, tests_config),
        DATASET_COLUMN_LOCAL_RECENCY_NAME,
    )
