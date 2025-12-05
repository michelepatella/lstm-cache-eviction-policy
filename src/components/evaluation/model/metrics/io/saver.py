"""saver.py

Utility module for saving model evaluation metrics.

This module provides the `save_model_metrics` function, which filters
and organizes a model's computed evaluation metrics, adds the average
loss, and saves the results as a JSON file to a specified path.

Functions:
    save_model_metrics(
        metrics: dict[str, int | float],
        avg_loss: float,
        path: str
    ) -> None
        Filters classification report, adds average loss, and saves
        metrics to a JSON file.
"""

import numpy as np

from components.const import (
    MODEL_METRICS_ACCURACY_NAME,
    MODEL_METRICS_AVG_LOSS_NAME,
    MODEL_METRICS_CLASS_REPORT_NAME,
    MODEL_METRICS_MACRO_AVG_NAME,
    MODEL_METRICS_WEIGHTED_AVG_NAME,
)
from components.json.io.saver import save_json
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def save_model_metrics(
    metrics: dict[str, int | float],
    avg_loss: float,
    path: str,
) -> None:
    """Save model metrics to a JSON file.

    This function filters the classification report to keep only
    desired metrics to be saved to a JSON file.

    Args:
        metrics (dict[str, int | float]): Dictionary of computed metrics.
        avg_loss (float): Average loss to include in the metrics.
        path (str): Path where the metrics JSON file should be saved.

    Returns:
        None

    Raises:
        RuntimeError: If saving the model metrics fails:
            * Invalid metric types (TypeError).
    """
    try:
        debug(
            "Model metrics saving started",
            extra={
                "metrics_num": len(metrics),
                "loss_avg": None
                if np.isinf(avg_loss) or np.isnan(avg_loss)
                else float(avg_loss),
                "save_path": str(path),
                "context": "Model metrics saving",
            },
        )

        # Filter metrics for saving
        metrics_to_save = {
            k: (
                {
                    rk: rv
                    for rk, rv in v.items()
                    if rk
                    in [
                        MODEL_METRICS_ACCURACY_NAME,
                        MODEL_METRICS_MACRO_AVG_NAME,
                        MODEL_METRICS_WEIGHTED_AVG_NAME,
                    ]
                }
                if k == MODEL_METRICS_CLASS_REPORT_NAME and isinstance(v, dict)
                else v
            )
            for k, v in metrics.items()
        }

        # Add average loss
        metrics_to_save[MODEL_METRICS_AVG_LOSS_NAME] = avg_loss

        # Save metrics as JSON file
        save_json(metrics_to_save, path)

        debug(
            "Model metrics saving completed",
            extra={
                "metrics_saved_num": len(metrics_to_save),
                "metrics_keys": list(metrics_to_save.keys()),
                "save_path": str(path),
                "context": "Model metrics saving",
            },
        )
    except TypeError as e:
        msg = "Model metrics saving failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "metrics_num": len(metrics) if metrics else 0,
                "loss_avg": None
                if np.isinf(avg_loss) or np.isnan(avg_loss)
                else float(avg_loss),
                "save_path": str(path),
                "context": "Model metrics saving",
            },
        )
        raise RuntimeError(msg) from e
