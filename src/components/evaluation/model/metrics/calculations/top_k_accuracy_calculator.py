import torch

from components.const import TENSOR_CLASS_DIM, TENSOR_OUTPUTS_BATCH_DIM
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.math.percentage_calculator import calculate_percentage


def calculate_top_k_accuracy(
    targets: list[int],
    outputs: list[torch.Tensor],
    top_k: int,
) -> float:
    """Compute top-k accuracy for a multi-class classification task.

    This function calculates the proportion of samples for which
    the true label is among the top-K predicted labels.

    Args:
        targets (list[int]): Ground truth class labels.
        outputs (list[torch.Tensor]): Model outputs.
        top_k (int): Number of top predictions to consider for accuracy.

    Returns:
        float: Top-k accuracy.

    Raises:
        RuntimeError: If top-k accuracy calculation fails:
            * Invalid output tensors (TypeError, RuntimeError).
            * Indexing errors while comparing predictions (IndexError).
            * Division by zero if no targets are provided (ZeroDivisionError).
    """
    try:
        # Stack outputs in a single 2D tensor
        outputs_tensor = torch.stack(outputs, dim=TENSOR_OUTPUTS_BATCH_DIM)

        # Extract top-k predictions along the class dimension
        top_k_predictions = (
            torch.topk(outputs_tensor, k=top_k, dim=TENSOR_CLASS_DIM)
            .indices.cpu()
            .numpy()
        )

        # Count correct predictions
        correct_predictions = sum(
            1
            for i, target in enumerate(targets)
            if target in top_k_predictions[i][:top_k]
        )

        # Compute top-k accuracy
        top_k_accuracy = calculate_percentage(
            correct_predictions,
            len(targets),
        ) / 100

        info(
            "Top-k accuracy calculated",
            extra={
                "targets_num": len(targets),
                "outputs_num": len(outputs),
                "top_k": top_k,
                "predictions_correct": correct_predictions,
                "top_k_accuracy": top_k_accuracy,
                "context": "Top-k accuracy calculation",
            },
        )

        return top_k_accuracy
    except (RuntimeError, IndexError, TypeError, ZeroDivisionError) as e:
        msg = "Top-k accuracy calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "targets_num": len(targets) if targets else 0,
                "outputs_num": len(outputs) if outputs else 0,
                "top_k": top_k,
                "context": "Top-k accuracy calculation",
            },
        )
        raise RuntimeError(msg) from e
