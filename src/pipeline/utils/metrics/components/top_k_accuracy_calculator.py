from typing import List

import torch

from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


def calculate_top_k_accuracy(
    targets: List[int],
    outputs: List[torch.Tensor],
    top_k: int,
) -> float:
    """
    Compute top-k accuracy for a multi-class classification task.

    This function calculates the proportion of samples for which
    the true label is among the top-K predicted labels.

    Args:
        targets (List[int]): Ground truth class labels.
        outputs (List[torch.Tensor]): Model outputs.
        top_k (int): Number of top predictions to consider for accuracy.

    Returns:
        float: Top-k accuracy.

    Raises:
        RuntimeError: If an error occurs while computing top-k accuracy, e.g.:
            * Outputs tensor has incompatible shape.
            * Top-k value is larger than number of classes.
            * Number of targets differs from number of predictions.
            * Targets or outputs are of incompatible type.
            * Targets list is empty.
    """
    try:
        debug(f"Targets length: {len(targets)}")
        debug(f"Outputs length: {len(outputs)}")

        # Stack outputs in a single 2D tensor
        outputs_tensor = torch.stack(outputs, dim=0)

        # Extract top-k predictions along the class dimension
        top_k_predictions = (
            torch.topk(outputs_tensor, k=top_k, dim=1).indices.cpu().numpy()
        )

        debug(f"Top-{top_k} predictions shape: {top_k_predictions.shape}")

        # Count correct predictions
        correct_predictions = 0
        for i, target in enumerate(targets):
            # Whether the current target is in the
            # top-k predicted keys
            if target in top_k_predictions[i][:top_k]:
                correct_predictions += 1
            debug(
                f"Sample {i}: target={target}, "
                f"top-k={top_k_predictions[i][:top_k]}, "
                f"correct={target in top_k_predictions[i][:top_k]}"
            )

        # Compute accuracy
        top_k_accuracy = correct_predictions / len(targets)

        info(f"Top-{top_k} accuracy calculated")
    except (RuntimeError, IndexError, TypeError, ZeroDivisionError) as e:
        msg = "Failed to compute top-k accuracy"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    return top_k_accuracy
