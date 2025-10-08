import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from const import (
    CONFUSION_MATRIX_PLOT_ANNOT,
    CONFUSION_MATRIX_PLOT_FMT,
    CONFUSION_MATRIX_PLOT_TITLE,
    CONFUSION_MATRIX_PLOT_X_LABEL,
    CONFUSION_MATRIX_PLOT_Y_LABEL,
    PLOT_LABEL_FONT_SIZE,
    PLOT_SIZE,
    PLOT_TITLE_FONT_SIZE,
    CONFUSION_MATRIX_PLOT_CMAP,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


def plot_confusion_matrix(confusion_matrix: dict, save_path: str) -> None:
    """
    Plot a confusion matrix.

    This function visualizes the provided confusion matrix
    as a heatmap.

    Args:
        confusion_matrix (dict): Confusion matrix to plot.
        save_path (str): Path to save the figure.

    Returns:
        None
    """
    debug(f"Confusion matrix shape: {np.array(confusion_matrix).shape}")
    debug(
        f"Confusion matrix min: {np.min(confusion_matrix)},"
        f" max: {np.max(confusion_matrix)}"
    )

    # Convert confusion matrix
    # to 2D numpy array
    labels = sorted(confusion_matrix.keys())
    confusion_matrix_array = np.array(
        [[confusion_matrix[i].get(j, 0) for j in labels] for i in labels]
    )

    # Normalize confusion matrix
    confusion_matrix_norm = (
        confusion_matrix_array / confusion_matrix_array.max()
    )

    # Plot confusion matrix
    plt.figure(figsize=(PLOT_SIZE, PLOT_SIZE))
    sns.heatmap(
        confusion_matrix_norm,
        annot=CONFUSION_MATRIX_PLOT_ANNOT,
        fmt=CONFUSION_MATRIX_PLOT_FMT,
        cmap=CONFUSION_MATRIX_PLOT_CMAP,
    )
    plt.title(CONFUSION_MATRIX_PLOT_TITLE, fontsize=PLOT_TITLE_FONT_SIZE)
    plt.xlabel(CONFUSION_MATRIX_PLOT_X_LABEL, fontsize=PLOT_LABEL_FONT_SIZE)
    plt.ylabel(CONFUSION_MATRIX_PLOT_Y_LABEL, fontsize=PLOT_LABEL_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()

    info(f"Confusion matrix plotted and saved to {save_path}")
