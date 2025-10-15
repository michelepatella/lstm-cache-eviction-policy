from typing import Dict, List

from pipeline.const import TIMELINE_NAME
from utils.json.saver import save_json
from utils.logs.levels.debug_logger import debug


def save_simulation_results(results: List[Dict], path: str) -> None:
    """
    Save cache simulation results to a JSON file.

    This function keeps only the simulation metrics
    to be saved, then stores the results as JSON.

    Args:
        results (List[Dict]): List of metrics dictionaries per policy.
        path (str): Path to save the JSON file.

    Returns:
        None
    """
    # Filter metrics to be saved
    results_to_save = [
        {k: v for k, v in result.items() if k != TIMELINE_NAME}
        for result in results
    ]

    # Save metrics to JSON file
    save_json(results_to_save, path)

    debug(f"Simulation results saved to {path}")
