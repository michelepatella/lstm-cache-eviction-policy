from typing import Dict

from pipeline.const import (
    MODEL_METRICS_ACCURACY_NAME,
    MODEL_METRICS_AVG_LOSS,
    MODEL_METRICS_CLASS_REPORT_NAME,
    MODEL_METRICS_MACRO_AVG_NAME,
    MODEL_METRICS_WEIGHTED_AVG_NAME,
)
from utils.json.saver import save_json
from utils.logs.levels.debug_logger import debug


def save_model_results(
    metrics: Dict[str, int | float], avg_loss: float, path: str
) -> None:
    """
    Save model metrics to a JSON file.

    This function filters the classification report to keep only
    desired metrics to be saved to a JSON file.

    Args:
        metrics (Dict[str, int | float]): Dictionary of computed metrics.
        avg_loss (float): Average loss to include in the metrics.
        path (str): Path where the metrics JSON file should be saved.

    Returns:
        None
    """
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
    metrics_to_save[MODEL_METRICS_AVG_LOSS] = avg_loss

    # Save metrics as JSON file
    save_json(metrics_to_save, path)

    debug(f"Model results saved to {path}")
