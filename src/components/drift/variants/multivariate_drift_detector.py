"""multivariate_drift_detector.py

This module implements multivariate drift detection using the Kolmogorov-Smirnov test.

It analyzes the joint distribution of all processed features to identify shifts
between historical reference data and new production data. The detection results
are stored as a JSON file.

Functions:
    detect_multivariate_drift(
        new_df: pd.DataFrame,
        hist_df: pd.DataFrame
    ) -> dict[str, Any]:
        Performs multivariate KS tests across all features and returns drift results.
"""

from typing import Any

import pandas as pd
from alibi_detect.cd import KSDrift
from box import Box

from components.const import (
    DATASET_PROCESSED_FEATURE_COLUMNS,
    DRIFT_DETECTION_MULTIVARIATE_RESULT_PATH,
    DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME,
    DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME,
    DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME,
)
from components.json.io.saver import save_json


def detect_multivariate_drift(
    new_df: pd.DataFrame,
    hist_df: pd.DataFrame,
) -> dict[str, Any]:
    """Detects multivariate drift across all features using the Kolmogorov-Smirnov test.

    This method creates a multivariate detector based on historical data and
    predicts drift on the latest production data. It aggregates p-values and
    distances for the entire feature set and saves the results to a JSON file.

    Args:
        new_df (pd.DataFrame): DataFrame containing the latest production data.
        hist_df (pd.DataFrame): DataFrame containing the historical reference data.

    Returns:
        dict[str, Any]: A dictionary containing the global drift status,
                        a list of p-values, and a list of distances for each feature.
    """
    # Create multivariate drift detector (Kolmogorov-Smirnov test)
    # on historical data
    ks_multivariate_detector = KSDrift(
        hist_df[DATASET_PROCESSED_FEATURE_COLUMNS].to_numpy(),
    )

    # Use multivariate KS detector on new data
    pred = Box(
        ks_multivariate_detector.predict(
            new_df[DATASET_PROCESSED_FEATURE_COLUMNS].to_numpy(),
        ),
    )

    # Save multivariate drift result
    multivariate_drift_results = {
        DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME: int(pred.data.is_drift),
        DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME: pred.data.p_val.tolist(),
        DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME: pred.data.distance.tolist(),
    }
    save_json(
        multivariate_drift_results,
        DRIFT_DETECTION_MULTIVARIATE_RESULT_PATH,
    )

    return multivariate_drift_results
