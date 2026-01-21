"""prediction_drift_detector.py

This module implements prediction drift detection using the Chi-Square test.

It monitors shifts in the distribution of the model's output (predictions) by
comparing predictions made on a historical baseline dataset against those made
on new production data.

Functions:
    detect_prediction_drift(
        new_dataloader: DataLoader,
        hist_dataloader: DataLoader,
        model: Module,
        device: torch.device
    ) -> dict[str, Any]:
        Performs a Chi-Square test on model predictions and returns drift results.
"""

from typing import Any

import torch
from alibi_detect.cd import ChiSquareDrift
from box import Box
from torch.nn import Module
from torch.utils.data import DataLoader

from components.const import (
    DRIFT_DETECTION_PREDICTION_RESULT_PATH,
    DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME,
    DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME,
    DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME,
)
from components.json.io.saver import save_json
from components.model.monitoring.utils.predictions_collector import (
    collect_model_predictions,
)


def detect_prediction_drift(
    new_dataloader: DataLoader,
    hist_dataloader: DataLoader,
    model: Module,
    device: torch.device,
) -> dict[str, Any]:
    """Detects drift in model predictions using the Chi-Square test.

    This method generates predictions for both historical and new datasets,
    compares their categorical distributions using a Chi-Square test, and
    determines if a significant shift in model behavior has occurred.
    Results are saved to a JSON file.

    Args:
        new_dataloader (DataLoader): DataLoader for the latest production data.
        hist_dataloader (DataLoader): DataLoader for the historical reference data.
        model (Module): The production model to evaluate.
        device (torch.device): The device selected for computation.

    Returns:
        dict[str, Any]: A dictionary containing the drift status, p-value,
                        and distance for the predictions.
    """
    prediction_drift_results = {}
    if model is not None:
        # Collect model predictions both
        # on new and historical data
        new_preds = collect_model_predictions(
            model,
            new_dataloader,
            device,
        )
        hist_preds = collect_model_predictions(
            model,
            hist_dataloader,
            device,
        )

        # Create Chi-Square as prediction drift detector
        # on historical predictions
        prediction_drift_detector = ChiSquareDrift(hist_preds)

        # Use Chi-Square on new predictions
        pred = Box(
            prediction_drift_detector.predict(new_preds),
        )

        # Save prediction drift results
        prediction_drift_results = {
            DRIFT_DETECTION_RESULT_FIELD_IS_DRIFT_NAME: int(
                pred.data.is_drift,
            ),
            DRIFT_DETECTION_RESULT_FIELD_P_VAL_NAME: float(pred.data.p_val),
            DRIFT_DETECTION_RESULT_FIELD_DISTANCE_NAME: float(
                pred.data.distance,
            ),
        }

    # Save overall results
    save_json(prediction_drift_results, DRIFT_DETECTION_PREDICTION_RESULT_PATH)

    return prediction_drift_results
