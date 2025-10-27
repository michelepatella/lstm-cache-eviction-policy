import torch
from scipy.stats import norm

from components.logs.levels.error_logger import error


def calculate_confidence_interval(
    outputs: list[torch.Tensor],
    variances: list[torch.Tensor],
    confidence_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate confidence interval for a set of outputs.

    This function calculates the lower and upper boundaries of the confidence
    interval for each element in the outputs, given their variances and a
    specified confidence level.

    Args:
        outputs (list[torch.Tensor]): List of output tensors.
        variances (list[torch.Tensor]): List of variance tensors corresponding
                                        to each output.
        confidence_level (float): Confidence level.

    Raises:
        RuntimeError: If calculation of confidence intervals fails:
            * If outputs or variances are not valid lists of torch.Tensor
              (TypeError).
            * If outputs and variances have different lengths (ValueError).
            * If confidence_level is not between 0 and 1 (ValueError).
            * If stacking tensors fails (RuntimeError).
    """
    try:
        # Calculate z-score for the
        # given confidence level
        z_score = norm.ppf(
            1 - (1 - confidence_level) / 2,
        )

        # Calculate standard deviation
        outputs_std = torch.sqrt(torch.stack(variances))

        outputs_tensor = torch.stack(outputs)
        # Calculate lower and upper CIs boundaries
        lower_ci = outputs_tensor - z_score * outputs_std
        upper_ci = outputs_tensor + z_score * outputs_std

        return lower_ci, upper_ci
    except (TypeError, ValueError, RuntimeError) as e:
        msg = "Confidence interval calculation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "outputs": outputs,
                "outputs_type": type(outputs).__name__,
                "variances": variances,
                "variances_type": type(variances).__name__,
                "confidence_level": confidence_level,
                "confidence_level_type": type(confidence_level).__name__,
                "context": "Confidence interval calculation",
            },
        )
        raise RuntimeError(msg) from e
