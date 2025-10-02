import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from const import (
    CONFUSION_MATRIX_PLOT_ANNOT,
    CONFUSION_MATRIX_PLOT_FMT,
    CONFUSION_MATRIX_PLOT_TITLE,
    CONFUSION_MATRIX_PLOT_X_LABEL,
    CONFUSION_MATRIX_PLOT_Y_LABEL,
    FIGURE_LABEL_FONT_SIZE,
    FIGURE_SIZE,
    FIGURE_TITLE_FONT_SIZE,
)
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.info_logger import info


def plot_confusion_matrix(confusion_matrix: dict) -> None:
    """
    Plot a confusion matrix.

    This function visualizes the provided confusion matrix
    as a heatmap.

    Parameters:
        confusion_matrix (dict): Confusion matrix to plot.

    Returns:
        None
    """
    debug(f"Confusion matrix shape: {np.array(confusion_matrix).shape}")
    debug(
        f"Confusion matrix min: {np.min(confusion_matrix)},"
        f" max: {np.max(confusion_matrix)}"
    )

    # Plot confusion matrix
    plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
    sns.heatmap(
        confusion_matrix,
        annot=CONFUSION_MATRIX_PLOT_ANNOT,
        fmt=CONFUSION_MATRIX_PLOT_FMT,
    )
    plt.title(CONFUSION_MATRIX_PLOT_TITLE, fontsize=FIGURE_TITLE_FONT_SIZE)
    plt.xlabel(CONFUSION_MATRIX_PLOT_X_LABEL, fontsize=FIGURE_LABEL_FONT_SIZE)
    plt.ylabel(CONFUSION_MATRIX_PLOT_Y_LABEL, fontsize=FIGURE_LABEL_FONT_SIZE)
    plt.tight_layout()
    plt.show()
    plt.close()

    info("Confusion matrix plotted")
