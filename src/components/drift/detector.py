import json
from datetime import datetime, timezone

import mlflow
import pandas as pd
import requests
import torch
from alibi_detect.cd import ChiSquareDrift, KSDrift
from box import Box

from api.const import MLFLOW_TRACKING_URI
from components.const import (
    DATASET_PROCESSED_FEATURE_COLUMNS,
    DATASET_REAL_PROCESSED_FILE_PATH,
    DRIFT_DETECTION_MULTIVARIATE_RESULT_PATH,
    DRIFT_DETECTION_PREDICTION_RESULT_PATH,
    DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME,
    DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME,
    DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME,
    DRIFT_DETECTION_TARGET_RESULT_PATH,
    DRIFT_DETECTION_UNIVARIATE_RESULT_PATH,
    LIST_LAST_IDX,
    LOGS_GRAFANA_LOKI_LOGS_URL,
    LOGS_GRAFANA_LOKI_TOKEN,
    LOGS_GRAFANA_LOKI_USER_ID,
    MODEL_EVAL_MODE,
    RETRAINING_CHECKPOINT_FILE_PATH,
    TIME_NANOSECONDS_IN_SECOND,
)
from components.data_loader.initializer import initialize_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.dataset.io.loader import load_dataset
from components.device.mover import move_to_device
from components.device.selector import select_device
from components.json.io.loader import load_json
from components.json.io.saver import save_json
from components.model.mode.setter import set_model_mode
from const import (
    DATASET_COLUMN_REQUEST_NAME,
    MLFLOW_MODEL_PRODUCTION_NAME,
    MLFLOW_MODEL_TAG_STATE,
    MLFLOW_MODEL_TAG_STATE_PROD,
)
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import DATASET_PROCESSED_TYPE


def _collect_predictions(model, dataloader, device):
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = move_to_device(x, device)
            outputs = model(x)

            y_hat = torch.argmax(outputs, dim=1)

            preds.append(y_hat.cpu())

    return torch.cat(preds).numpy()


def detect_drift() -> bool:
    # ----------------------------
    # Setup
    # ----------------------------
    # Prepare pipeline configuration
    pipeline_config = prepare_pipeline_config()

    # Load retraining configuration
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

    # Prepare new and historical dataloaders
    _, new_dataloader = initialize_data_loader(
        DATASET_PROCESSED_TYPE,
        None,
        pipeline_config.data_loader.testing.batch_size,
        pipeline_config.data_loader.testing.shuffle,
        AccessLogsDataset,
        pipeline_config,
    )
    _, hist_dataloader = initialize_data_loader(
        DATASET_PROCESSED_TYPE,
        None,
        pipeline_config.data_loader.testing.batch_size,
        pipeline_config.data_loader.testing.shuffle,
        AccessLogsDataset,
        pipeline_config,
    )

    # Set quantization engine
    torch.backends.quantized.engine = (
        pipeline_config.model.optimizations.quantization.engine
    )

    # Load the last version of the production model
    model = None
    mlflow_client = mlflow.MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI,
    )
    model_versions = mlflow_client.search_model_versions(
        f"name='{MLFLOW_MODEL_PRODUCTION_NAME}'",
    )
    prod_versions = [
        v
        for v in model_versions
        if v.tags.get(MLFLOW_MODEL_TAG_STATE) == MLFLOW_MODEL_TAG_STATE_PROD
    ]
    last_model_version = max(
        (v for v in prod_versions),
        key=lambda v: int(v.version),
        default=None,
    )
    if last_model_version is not None:
        model = mlflow.pytorch.load_model(
            model_uri=f"models:/{MLFLOW_MODEL_PRODUCTION_NAME}/{last_model_version.version}",
        )

    # Move model to device
    device = select_device(pipeline_config.resources.devices.testing)
    model = move_to_device(model, device)

    # Set model to evaluation mode
    set_model_mode(model, MODEL_EVAL_MODE)

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

    # Save results
    save_json(univariate_drift_results, DRIFT_DETECTION_UNIVARIATE_RESULT_PATH)

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

    # Save results
    save_json(
        multivariate_drift_results,
        DRIFT_DETECTION_MULTIVARIATE_RESULT_PATH,
    )

    # ----------------------------
    # Target Drift Detection
    # ----------------------------
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

    # Save results
    save_json(target_drift_results, DRIFT_DETECTION_TARGET_RESULT_PATH)

    # ----------------------------
    # Prediction Drift Detection
    # ----------------------------
    prediction_drift_results = {}
    if model is not None:
        # Collect predictions
        hist_preds = _collect_predictions(
            model,
            hist_dataloader,
            device,
        )
        new_preds = _collect_predictions(
            model,
            new_dataloader,
            device,
        )

        # Chi-Square drift on predictions
        prediction_drift_detector = ChiSquareDrift(hist_preds)
        pred = Box(
            prediction_drift_detector.predict(new_preds),
        )

        # Save prediction drift results
        prediction_drift_results = {
            DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME: pred.data.is_drift,
            DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME: pred.data.p_val,
            DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME: pred.data.distance,
        }

    # Save results
    save_json(prediction_drift_results, DRIFT_DETECTION_PREDICTION_RESULT_PATH)

    # ----------------------------
    # Final checking
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
        or target_drift_results.get(DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME)
        or prediction_drift_results.get(
            DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME,
        )
    ):
        return True
    return False


if __name__ == "__main__":
    detect_drift()
