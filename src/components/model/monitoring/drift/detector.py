"""detector.py

Module responsible for orchestrating the whole drift detection workflow.

This module provides the `detect_drift` function, which acts as the main entry
point for monitoring data stability. It coordinates the execution of specialized
drift detectors (univariate, multivariate, target, and prediction), aggregates
their results, and returns a boolean flag indicating whether any form of drift
has been detected.

Functions:
    detect_drift() -> bool:
        Orchestrates specialized drift checks and returns the final detection status.
"""

from components.const import DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME
from components.model.monitoring.drift.variants.multivariate_drift_detector import (
    detect_multivariate_drift,
)
from components.model.monitoring.drift.variants.prediction_drift_detector import (
    detect_prediction_drift,
)
from components.model.monitoring.drift.variants.target_drift_detector import (
    detect_target_drift,
)
from components.model.monitoring.drift.variants.univariate_drift_detector import (
    detect_univariate_drift,
)
from components.model.monitoring.initializer import initialize_model_monitoring


def detect_drift() -> bool:
    """Orchestrates the detection of data, target, and prediction drift.

    This function performs the following steps:
    1.  Setup: Initializes the drift detection environment, retrieving new
        production data (DataFrame and DataLoader), historical reference data,
        and the current production model.
    2.  Univariate Drift: Checks for distribution shifts in individual features.
    3.  Multivariate Drift: Checks for shifts in the joint distribution of features.
    4.  Target Drift: Checks for shifts in the ground truth labels.
    5.  Prediction Drift: Checks for shifts in the model's output distribution.
    6.  Final Checking: Evaluates all results and returns True if any specific
        detector signals that drift has occurred.

    Returns:
        bool: True if any form of drift is detected, False otherwise.
    """
    # ----------------------------
    # Setup
    # ----------------------------
    (
        new_df,
        hist_df,
        _,
        _,
        new_dataloader,
        hist_dataloader,
        pipeline_config,
        model,
        device,
    ) = initialize_model_monitoring()

    # ----------------------------
    # Preconditions
    # ----------------------------
    # Check if enough data is available
    if len(new_df) < pipeline_config.training.samples.min:
        return False

    # ----------------------------
    # Univariate Drift Detection
    # ----------------------------
    univariate_drift_results = detect_univariate_drift(new_df, hist_df)

    # ----------------------------
    # Multivariate Drift Detection
    # ----------------------------
    multivariate_drift_results = detect_multivariate_drift(new_df, hist_df)

    # ----------------------------
    # Target Drift Detection
    # ----------------------------
    target_drift_results = detect_target_drift(new_df, hist_df)

    # ----------------------------
    # Prediction Drift Detection
    # ----------------------------
    prediction_drift_results = detect_prediction_drift(
        new_dataloader,
        hist_dataloader,
        model,
        device,
    )

    # ----------------------------
    # Final Checking
    # ----------------------------
    # Return True if any drift detected, False otherwise
    if (
        any(
            res[DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME]
            for res in univariate_drift_results.values()
        )
        or multivariate_drift_results[
            DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME
        ]
        or target_drift_results[DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME]
        or prediction_drift_results[DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME]
    ):
        return True
    return False


if __name__ == "__main__":
    detect_drift()
