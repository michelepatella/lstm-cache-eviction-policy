"""saver.py

Utility module for saving cache simulation metrics.

This module provides the `save_simulations_metrics` function, which
filters and stores cache simulation metrics in JSON format. It ensures
that only relevant metrics are saved, excluding detailed timelines.

Functions:
    save_simulations_metrics(
        results: list[dict],
        path: str
    ) -> None
        Saves the filtered simulation metrics to the specified JSON file.
"""

from components.json.io.saver import save_json
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from const import SIMULATIONS_METRICS_TIMELINE_NAME


def save_simulations_metrics(results: list[dict], path: str) -> None:
    """Save cache simulations metrics to a JSON file.

    This function keeps only the simulations metrics
    to be saved, then stores the results as JSON.

    Args:
        results (list[dict]): List of metrics dictionaries per policy.
        path (str): Path to save the JSON file.

    Returns:
        None

    Raises:
        RuntimeError: If saving simulation metrics fails:
            * Provided results are not a list of dictionaries (TypeError).
            * File cannot be written due to OS errors (OSError).
    """
    try:
        debug(
            "Simulations metrics saving started",
            extra={
                "results_num": (
                    len(results) if isinstance(results, list) else None
                ),
                "save_path": str(path),
                "context": "Simulations metrics saving",
            },
        )

        # Filter metrics to be saved
        results_to_save = [
            {
                k: v
                for k, v in result.items()
                if k != SIMULATIONS_METRICS_TIMELINE_NAME
            }
            for result in results
        ]

        # Save metrics to JSON file
        save_json(results_to_save, path)

        debug(
            "Simulations metrics saving completed",
            extra={
                "results_saved_num": len(results_to_save),
                "save_path": str(path),
                "context": "Simulations metrics saving",
            },
        )
    except (TypeError, OSError) as e:
        msg = "Simulations metrics saving failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "results_num": (
                    len(results) if isinstance(results, list) else None
                ),
                "save_path": str(path),
                "context": "Simulations metrics saving",
            },
        )
        raise RuntimeError(msg) from e
