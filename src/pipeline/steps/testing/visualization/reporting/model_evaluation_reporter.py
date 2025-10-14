from box.box import Box

from utils.logs.levels.info_logger import info


def generate_model_evaluation_report(
    class_report: dict,
    top_k_accuracy: float,
    kappa_statistic: float,
    avg_loss: float,
    top_k: int,
) -> None:
    """
    Generate a report for model evaluation
    results.

    This function reports model evaluation results via
    informational logs.

    Args:
        class_report (dict): Classification metrics per class.
        top_k_accuracy (float): Top-K accuracy of the model.
        kappa_statistic (float): Cohen's kappa statistic of the model.
        avg_loss (float): Average loss of the model.
        top_k (int): Value of K used for top-k accuracy of the model.

    Returns:
        None
    """
    # Box for class report
    class_report = Box(class_report)

    # General metrics
    info("=== LSTM Standalone Evaluation Report ===")
    info(f"Average Loss      : {avg_loss}")
    info(f"Top-{top_k} Accuracy: {top_k_accuracy}")
    info(f"Kappa Statistic   : {kappa_statistic}")

    # Per-class metrics
    for cls in sorted(
        (k for k in class_report.keys() if k.isdigit()), key=int
    ):
        metrics = class_report[cls]
        info(
            f"Class {cls}: Precision={metrics.precision}, "
            f"Recall={metrics.recall}, F1={metrics.f1_score}, "
            f"Support={int(metrics.support)}"
        )

    # Summary metrics
    macro = class_report.macro_avg
    weighted = class_report.weighted_avg
    info(
        f"Macro Avg: Precision={macro.precision}, "
        f"Recall={macro.recall}, F1={macro.f1_score}"
    )
    info(
        f"Weighted Avg: Precision={weighted.precision}, "
        f"Recall={weighted.recall}, F1={weighted.f1_score}"
    )
    info(f"Accuracy: {class_report.accuracy}")
    info("=== End of Report ===")
