from typing import Dict, List

from components.json.io.saver import save_json
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from const import TIMELINE_NAME


def save_simulations_metrics(results: List[Dict], path: str) -> None:
    """
    Save cache simulations metrics to a JSON file.

    This function keeps only the simulations metrics
    to be saved, then stores the results as JSON.

    Args:
        results (List[Dict]): List of metrics dictionaries per policy.
        path (str): Path to save the JSON file.

    Returns:
        None

    Raises:
        RuntimeError: If saving simulation metrics fails:
            * Provided results are not a list of dictionaries (TypeError).
            * File cannot be written due to OS errors (OSError).
    """
    try:
        # Filter metrics to be saved
        results_to_save = [
            {k: v for k, v in result.items() if k != TIMELINE_NAME}
            for result in results
        ]

        # Save metrics to JSON file
        save_json(results_to_save, path)

        info(f"Simulations results saved to: {path}")
    except (TypeError, OSError) as e:
        msg = "Failed to save simulation metrics"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
