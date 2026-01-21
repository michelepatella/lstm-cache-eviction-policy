"""univariate_drift_detector.py

This module implements univariate drift detection using the Kolmogorov-Smirnov test.

It iterates through individual features of the dataset to identify shifts in
their distributions between historical reference data and new production data.
The results are stored as a JSON file for further analysis.

Functions:
    detect_univariate_drift(
        new_df: pd.DataFrame,
        hist_df: pd.DataFrame
    ) -> dict[str, Any]:
        Performs KS tests on individual features and returns the drift results.
"""

from typing import Any

import pandas as pd
from alibi_detect.cd import KSDrift
from box import Box

from components.const import (
    DATASET_PROCESSED_FEATURE_COLUMNS,
    DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME,
    DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME,
    DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME,
    DRIFT_DETECTION_UNIVARIATE_RESULT_PATH,
)
from components.json.io.saver import save_json


def detect_univariate_drift(
    new_df: pd.DataFrame,
    hist_df: pd.DataFrame,
) -> dict[str, Any]:
    """Detects univariate drift for each feature using the Kolmogorov-Smirnov test.

    This method compares the distribution of each processed feature in the new
    dataframe against the historical distribution. It calculates the p-value
    and distance, determines if drift is present, and saves the aggregate
    results to a local JSON file.

    Args:
        new_df (pd.DataFrame): DataFrame containing the latest production data.
        hist_df (pd.DataFrame): DataFrame containing the historical reference data.

    Returns:
        dict[str, Any]: A dictionary where keys are feature names and values are
                        dictionaries containing drift status, p-value, and distance.
    """
    univariate_drift_results = {}
    for feature in DATASET_PROCESSED_FEATURE_COLUMNS:
        # Create univariate drift detector (Kolmogorov-Smirnov test)
        # on historical data
        ks_univariate_detector = KSDrift(hist_df[feature].to_numpy())

        # Use univariate KS detector on new data
        pred = Box(ks_univariate_detector.predict(new_df[feature].to_numpy()))

        # Save univariate drift results individually
        univariate_drift_results[feature] = {
            DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME: int(
                pred.data.is_drift,
            ),
            DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME: float(pred.data.p_val),
            DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME: float(
                pred.data.distance,
            ),
        }

    # Save overall results
    save_json(univariate_drift_results, DRIFT_DETECTION_UNIVARIATE_RESULT_PATH)

    return univariate_drift_results
