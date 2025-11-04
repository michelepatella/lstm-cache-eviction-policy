"""hit_miss_rates_plotter.py

Utility module for plotting the evolution of cache hit and miss rates
across multiple simulation runs and cache policies.

This module provides the `plot_hit_miss_rate` function, which generates
a multi-subplot figure illustrating how the hit rate and miss rate change
over time for various cache eviction strategies.

Functions:
    plot_hit_miss_rate(
        results: list[dict[str, Any]],
        path: str
    ) -> None
        Generates and saves a figure with subplots for hit rate and
        miss rate evolution over time.
"""

from typing import Any

import matplotlib.pyplot as plt

from components.const import (
    PLOT_HIT_MISS_RATE_SUBPLOT_X_LABEL,
    PLOT_HIT_MISS_RATES_NUM_COLS,
    PLOT_HIT_MISS_RATES_NUM_ROWS,
    PLOT_HIT_MISS_RATES_PAD,
    PLOT_HIT_MISS_RATES_SUBPLOTS,
    PLOT_HIT_MISS_RATES_SUBPLOTS_LINE_STYLE_NAME,
    PLOT_HIT_MISS_RATES_SUBPLOTS_TITLE_NAME,
    PLOT_HIT_MISS_RATES_SUBPLOTS_TRANSFORM_NAME,
    PLOT_HIT_MISS_RATES_SUBPLOTS_Y_LABEL_NAME,
    PLOT_LABEL_FONT_SIZE,
    PLOT_SIZE,
    PLOT_TITLE_FONT_SIZE,
    SIMULATIONS_METRICS_TIMELINE_INDEX_NAME,
    SIMULATIONS_METRICS_TIMELINE_INSTANT_HIT_RATE_NAME,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from const import (
    SIMULATIONS_METRICS_POLICY_NAME,
    SIMULATIONS_METRICS_TIMELINE_NAME,
)


def plot_hit_miss_rate(
    results: list[dict[str, Any]],
    path: str,
) -> None:
    """Plot the evolution of hit and miss rates over time for multiple
    cache eviction policies.

    This function generates two subplots: one for hit rate and one for
    miss rate. The generated subplots show the evolution of hit and miss
    rates over time across different cache policies.

    Args:
        results (list[dict[str, Any]]): List of simulations results.
        path (str): Path where to save the figure.

    Returns:
        None

    Raises:
        RuntimeError: If plotting the hit and miss rates fails:
            * Results or timeline entries are not of expected type (TypeError).
            * Required keys are missing (KeyError).
            * Data shapes or plotting values are invalid (ValueError).
            * matplotlib objects do not have expected attributes (AttributeError).
            * Subplot axes indexing fails (IndexError).
    """
    try:
        # Setup for the whole plot
        fig, axes = plt.subplots(
            nrows=PLOT_HIT_MISS_RATES_NUM_ROWS,
            ncols=PLOT_HIT_MISS_RATES_NUM_COLS,
            figsize=(PLOT_SIZE, PLOT_SIZE),
        )
        fig.tight_layout(pad=PLOT_HIT_MISS_RATES_PAD)

        # Ensure axes is always a list
        # (even of a single value)
        if PLOT_HIT_MISS_RATES_NUM_ROWS * PLOT_HIT_MISS_RATES_NUM_COLS == 1:
            axes = [axes]
        else:
            axes = list(axes)

        # Plot both subplots
        for i, subplot in enumerate(PLOT_HIT_MISS_RATES_SUBPLOTS):
            # Iterate over cache eviction policies
            for result in results:
                # Extract policy name and its timeline
                policy = result[SIMULATIONS_METRICS_POLICY_NAME]
                timeline = result[SIMULATIONS_METRICS_TIMELINE_NAME]

                # Extract x and y points
                x = [
                    point[SIMULATIONS_METRICS_TIMELINE_INDEX_NAME]
                    for point in timeline
                ]
                y = [
                    subplot[PLOT_HIT_MISS_RATES_SUBPLOTS_TRANSFORM_NAME](
                        point[
                            SIMULATIONS_METRICS_TIMELINE_INSTANT_HIT_RATE_NAME
                        ],
                    )
                    for point in timeline
                ]

                # Plot x and y points
                axes[i].plot(
                    x,
                    y,
                    label=policy,
                    linestyle=subplot[
                        PLOT_HIT_MISS_RATES_SUBPLOTS_LINE_STYLE_NAME
                    ],
                )

            # Set title and labels
            axes[i].set_title(
                subplot[PLOT_HIT_MISS_RATES_SUBPLOTS_TITLE_NAME],
                fontsize=PLOT_TITLE_FONT_SIZE,
            )
            axes[i].set_xlabel(
                PLOT_HIT_MISS_RATE_SUBPLOT_X_LABEL,
                fontsize=PLOT_LABEL_FONT_SIZE,
            )
            axes[i].set_ylabel(
                subplot[PLOT_HIT_MISS_RATES_SUBPLOTS_Y_LABEL_NAME],
                fontsize=PLOT_LABEL_FONT_SIZE,
            )
            axes[i].legend()

        # Show and save plot
        plt.savefig(path)
        plt.show()
        plt.close(fig)

        debug(
            "Hit/miss rates plotted and saved",
            extra={
                "save_path": str(path),
                "subplots_num": len(PLOT_HIT_MISS_RATES_SUBPLOTS),
                "results_num": len(results),
                "policies": [
                    r[SIMULATIONS_METRICS_POLICY_NAME] for r in results
                ],
                "context": "Hit/miss rates plot",
            },
        )
    except (TypeError, KeyError, ValueError, AttributeError, IndexError) as e:
        msg = "Plotting hit/miss rates failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "save_path": str(path),
                "subplots_num": len(PLOT_HIT_MISS_RATES_SUBPLOTS),
                "results_num": (
                    len(results) if isinstance(results, list) else None
                ),
                "policies": (
                    [r.get(SIMULATIONS_METRICS_POLICY_NAME) for r in results]
                    if isinstance(results, list)
                    else None
                ),
                "context": "Hit/miss rates plot",
            },
        )
        raise RuntimeError(msg) from e
