"""initializer.py

Module responsible for initializing the environment required for drift detection.

This module handles the retrieval of new production data from Grafana Loki, loads
historical reference data, and sets up the necessary PyTorch components (model,
data loaders, device) to perform drift analysis comparing current production data
against training data.

Functions:
    initialize_drift_detection() -> tuple[
        DataFrame,
        DataFrame,
        DataLoader,
        DataLoader,
        Module | Tensor,
        device
    ]
        Prepares datasets, loaders, and the production model for drift checking.
"""

import json
from datetime import datetime, timezone

import mlflow
import pandas as pd
import requests
import torch
from pandas import DataFrame
from torch import Tensor, device
from torch.nn import Module
from torch.utils.data import DataLoader

from api.const import MLFLOW_TRACKING_URI
from components.const import (
    DATASET_REAL_PROCESSED_FILE_PATH,
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
from components.model.mode.setter import set_model_mode
from const import (
    MLFLOW_MODEL_PRODUCTION_NAME,
    MLFLOW_MODEL_TAG_STATE,
    MLFLOW_MODEL_TAG_STATE_PROD,
)
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import DATASET_PROCESSED_TYPE


def initialize_drift_detection() -> tuple[
    DataFrame,
    DataFrame,
    DataLoader,
    DataLoader,
    Module | Tensor,
    device,
]:
    """Initializes the environment and data for drift detection.

    This function orchestrates the data retrieval and system setup phase by
    gathering and preparing production data, loading historical dataset,
    initializing dataloaders, and loading and preparing model environment.

    Returns:
        tuple[
            DataFrame,
            DataFrame,
            DataLoader,
            DataLoader,
            Module | Tensor,
            device
        ]:
            - new_df: DataFrame containing recent production data fetched from Loki.
            - hist_df: DataFrame containing historical reference data.
            - new_dataloader: DataLoader for the new production data.
            - hist_dataloader: DataLoader for the historical reference data.
            - model: The PyTorch model loaded from MLflow (production version).
            - device: The computing device selected for operations.
    """
    # Prepare pipeline configuration
    pipeline_config = prepare_pipeline_config()

    # Load retraining checkpoint
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

    # Prepare new and historical data loaders
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

    return new_df, hist_df, new_dataloader, hist_dataloader, model, device
