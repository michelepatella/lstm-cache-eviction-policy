import json
from datetime import datetime, timezone

import pandas as pd
import requests
from alibi_detect.cd import ChiSquareDrift, KSDrift
from box import Box

from components.const import (
    DATASET_PROCESSED_FEATURE_COLUMNS,
    DATASET_REAL_PROCESSED_FILE_PATH,
    DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME,
    DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME,
    DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME,
    LIST_LAST_IDX,
    LOGS_GRAFANA_LOKI_LOGS_URL,
    LOGS_GRAFANA_LOKI_TOKEN,
    LOGS_GRAFANA_LOKI_USER_ID,
    RETRAINING_CHECKPOINT_FILE_PATH,
    TIME_NANOSECONDS_IN_SECOND,
)
from components.dataset.io.loader import load_dataset
from components.json.io.loader import load_json
from const import DATASET_COLUMN_REQUEST_NAME


def detect_drift() -> bool:
    # ----------------------------
    # Setup
    # ----------------------------
    # Load retraining config
    retraining_checkpoint = load_json(RETRAINING_CHECKPOINT_FILE_PATH)

    # Get current timestamp in ns
    current_timestamp = int(
        datetime.now(timezone.utc).timestamp() * TIME_NANOSECONDS_IN_SECOND,
    )

    # Query Loki logs
    response = requests.get(
        LOGS_GRAFANA_LOKI_LOGS_URL,
        auth=(LOGS_GRAFANA_LOKI_USER_ID, LOGS_GRAFANA_LOKI_TOKEN),
        params={
            "query": '{service_name="unknown_service"} | json | context="Real-world data"',
            "start": retraining_checkpoint.last_timestamp + 1,
            "end": current_timestamp,
        },
    ).json()

    # Flatten data into records resulting in new dataset
    new_df = pd.DataFrame(
        [
            item
            for entry in response.get("data", {}).get("result", [])
            for row in entry["values"]
            for item in json.loads(row[LIST_LAST_IDX])["data"]
        ],
    )

    # Load historical dataset
    hist_df = load_dataset(DATASET_REAL_PROCESSED_FILE_PATH)

    # ----------------------------
    # Univariate Drift Detection
    # ----------------------------
    univariate_drift_results = {}
    for feature in DATASET_PROCESSED_FEATURE_COLUMNS:
        # Create univariate drift detector (Kolmogorov-Smirnov test)
        ks_univariate_detector = KSDrift(hist_df[feature].to_numpy())

        # Use univariate KS detector on new data
        pred = Box(ks_univariate_detector.predict(new_df[feature].to_numpy()))

        # Save univariate drift results
        univariate_drift_results[feature] = {
            DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME: pred.data.is_drift,
            DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME: pred.data.p_val,
            DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME: pred.data.distance,
        }

    # ----------------------------
    # Multivariate Drift Detection
    # ----------------------------
    # Create multivariate drift detector (Kolmogorov-Smirnov test)
    ks_multivariate_detector = KSDrift(
        hist_df[DATASET_PROCESSED_FEATURE_COLUMNS].to_numpy(),
    )

    # Use multivariate KS detector on new data
    pred = Box(
        ks_multivariate_detector.predict(
            new_df[DATASET_PROCESSED_FEATURE_COLUMNS].to_numpy(),
        ),
    )

    # Save multivariate drift results
    multivariate_drift_results = {
        DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME: pred.data.is_drift,
        DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME: pred.data.p_val,
        DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME: pred.data.distance,
    }

    # ----------------------------
    # Target Drift Detection
    # ----------------------------
    target_drift_results = {}
    if (
        DATASET_COLUMN_REQUEST_NAME in hist_df.columns
        and DATASET_COLUMN_REQUEST_NAME in new_df.columns
    ):
        # Extract target column both from historical and new datasets
        hist_target = hist_df[DATASET_COLUMN_REQUEST_NAME].to_numpy()
        new_target = new_df[DATASET_COLUMN_REQUEST_NAME].to_numpy()

        # Create target drift detector (Chi-Square)
        target_detector = ChiSquareDrift(hist_target)

        # Use target detector on new data
        pred = Box(target_detector.predict(new_target))

        # Save target drift results
        target_drift_results = {
            DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME: pred.data.is_drift,
            DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME: pred.data.p_val,
            DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME: pred.data.distance,
        }

    # TODO: Save drift detection results to 'reports'

    # Return True if any drift detected, False otherwise
    if (
        any(
            res[DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME]
            for res in univariate_drift_results.values()
        )
        or any(
            res[DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME]
            for res in multivariate_drift_results.values()
        )
        or any(
            res[DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME]
            for res in target_drift_results.values()
        )
    ):
        return True
    return False


if __name__ == "__main__":
    detect_drift()
