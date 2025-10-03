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

    Parameters:
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
    info("=== Model Evaluation Report ===")
    info(f"Average Loss      : {avg_loss:.4f}")
    info(f"Top-{top_k} Accuracy: {top_k_accuracy:.4f}")
    info(f"Kappa Statistic   : {kappa_statistic:.4f}")

    # Per-class metrics
    for cls in sorted(
        (k for k in class_report.keys() if k.isdigit()), key=int
    ):
        metrics = class_report[cls]
        info(
            f"Class {cls}: Precision={metrics.precision:.4f}, "
            f"Recall={metrics.recall:.4f}, F1={metrics.f1_score:.4f}, "
            f"Support={int(metrics.support)}"
        )

    # Summary metrics
    macro = class_report.macro_avg
    weighted = class_report.weighted_avg
    info(
        f"Macro Avg: Precision={macro.precision:.4f}, "
        f"Recall={macro.recall:.4f}, F1={macro.f1_score:.4f}"
    )
    info(
        f"Weighted Avg: Precision={weighted.precision:.4f}, "
        f"Recall={weighted.recall:.4f}, F1={weighted.f1_score:.4f}"
    )
    info(f"Accuracy: {class_report.accuracy:.4f}")
    info("=== End of Report ===")
