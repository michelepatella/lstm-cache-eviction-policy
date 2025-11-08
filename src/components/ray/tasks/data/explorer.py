"""explorer.py

Module defining Ray remote tasks for parallelized data exploration.

This module wraps plotting functions (Zipf distribution, daily profile,
and key usage heatmap) as Ray remote functions. This allows for the simultaneous
and asynchronous generation and saving of explorative plots, improving the
efficiency of the data exploration process.

Functions:
    plot_zipf_loglog_task(requests: list[int], save_path: str) -> None
        Remote task to plot the Zipf log-log distribution.
    plot_daily_profile_taskplot_daily_profile_task(
        timestamps_hours: np.ndarray,
        save_path: str
    ) -> None
        Remote task to plot the request daily profile.
    plot_key_usage_heatmap_taskplot_key_usage_heatmap_task(
        requests: list[int],
        timestamps_hours: np.ndarray,
        min_key: int,
        max_key: int,
        save_path: str,
    ) -> None
        Remote task to plot the key usage heatmap over time.
"""

import numpy as np
import ray

from components.visualization.daily_profile_plotter import plot_daily_profile
from components.visualization.key_usage_heatmap_plotter import (
    plot_key_usage_heatmap,
)
from components.visualization.zipf_loglog_plotter import plot_zipf_loglog


@ray.remote
def plot_zipf_loglog_task(requests: list[int], save_path: str) -> None:
    """Remote task to generate and save the Zipf log-log plot.

    This function is executed remotely via Ray. It calls the underlying
    plotting utility and returns the path where the resulting plot was saved.

    Args:
        requests (list[int]): A list of requested keys.
        save_path (str): The path where the plot should be saved.

    Returns:
        None
    """
    plot_zipf_loglog(requests, save_path)


@ray.remote
def plot_daily_profile_task(
    timestamps_hours: np.ndarray,
    save_path: str,
) -> None:
    """Remote task to generate and save the daily profile plot.

    This function is executed remotely via Ray. It visualizes the temporal
    distribution of requests over a 24-hour cycle and returns the save path.

    Args:
        timestamps_hours (np.ndarray): An array of request timestamps (in hours).
        save_path (str): The path where the plot should be saved.

    Returns:
        None
    """
    plot_daily_profile(timestamps_hours, save_path)


@ray.remote
def plot_key_usage_heatmap_task(
    requests: list[int],
    timestamps_hours: np.ndarray,
    min_key: int,
    max_key: int,
    save_path: str,
) -> None:
    """Remote task to generate and save the key usage heatmap plot.

    This function is executed remotely via Ray. It visualizes the
    frequency of access for each key across time and returns the save path.

    Args:
        requests (list[int]): A list of requested keys.
        timestamps_hours (np.ndarray): An array of request timestamps (in hours).
        min_key (int): The minimum key.
        max_key (int): The maximum key.
        save_path (str): The path where the plot should be saved.

    Returns:
        None
    """
    plot_key_usage_heatmap(
        min_key,
        max_key,
        requests,
        timestamps_hours,
        save_path,
    )
