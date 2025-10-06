from typing import List

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

from const import (
    FIGURE_LABEL_FONT_SIZE,
    FIGURE_TITLE_FONT_SIZE,
    PRECISION_RECALL_CURVE_CLASS_LABEL,
    PRECISION_RECALL_CURVE_TITLE,
    PRECISION_RECALL_CURVE_X_LABEL,
    PRECISION_RECALL_CURVE_Y_LABEL,
    PLOTS_SAVE_PATH,
    PRECISION_RECALL_CURVE_SAVE_PATH_NAME,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


def plot_precision_recall_curve(
    targets: List[int],
    outputs: np.ndarray,
    num_keys: int,
    data_distribution_mode: str,
) -> None:
    """
    Plot precision-recall curves.

    This function plots precision-recall curves
    for a multi-class classification problem, given
    all the targets and the model predictions.

    Parameters:
        targets (List[int]): True class labels (1D array).
        outputs (np.ndarray): Model outputs/probabilities (2D array).
        num_keys (int): Number of keys (i.e., classes).
        data_distribution_mode (str): Data distribution mode.

    Returns:
        None
    """
    debug(
        f"Targets and outputs length for "
        f"precision-recall curve: {len(targets)}, {len(outputs)}"
    )
    debug(f"Number of keys for precision-recall curve: {num_keys}")

    # Convert targets to one-hot representation
    targets_bin = label_binarize(targets, classes=np.arange(num_keys))

    # Plot one-vs-rest precision-recall curves
    # for each class
    for i in range(num_keys):
        # Get precision and recall for the
        # current class
        precision, recall, _ = precision_recall_curve(
            targets_bin[:, i], outputs[:, i]
        )

        # Plot precision and recall
        # for the current class
        plt.plot(
            recall,
            precision,
            label=f"{PRECISION_RECALL_CURVE_CLASS_LABEL} {i}",
        )

    plt.title(PRECISION_RECALL_CURVE_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)
    plt.xlabel(PRECISION_RECALL_CURVE_X_LABEL, fontsize=FIGURE_LABEL_FONT_SIZE)
    plt.ylabel(PRECISION_RECALL_CURVE_Y_LABEL, fontsize=FIGURE_LABEL_FONT_SIZE)
    plt.tight_layout()

    save_path = (
        PLOTS_SAVE_PATH
        / data_distribution_mode
        / PRECISION_RECALL_CURVE_SAVE_PATH_NAME
    )
    plt.savefig(save_path)

    plt.show()
    plt.close()

    info(f"Precision-recall curve plotted and saved to {save_path}")
