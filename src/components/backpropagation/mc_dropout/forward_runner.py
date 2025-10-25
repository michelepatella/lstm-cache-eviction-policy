from typing import Optional, Tuple, Union

import torch

from components.backpropagation.core.forward_runner import compute_forward
from components.const import (
    MC_DROPOUT_FLAG_NAME,
    MC_DROPOUT_NUM_SAMPLES_DEFAULT,
    MC_DROPOUT_UNBIASED_VARIANCE_DISABLED,
    MODEL_EVALUATION_MODE,
    MODEL_MC_DROPOUT_MODE,
)
from components.logs.levels.error_logger import error
from components.model.mode.setter import set_model_mode


def compute_mc_dropout_forward(
    model: torch.nn.Module,
    batch: Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ],
    device: torch.device,
    num_features: int,
    num_mc_dropout_samples: int = MC_DROPOUT_NUM_SAMPLES_DEFAULT,
    mc_dropout_flag: str = MC_DROPOUT_FLAG_NAME,
    mc_dropout_unbiased_variance=MC_DROPOUT_UNBIASED_VARIANCE_DISABLED,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Perform forward pass with optional Monte Carlo (MC) Dropout.

    This function executes forward pass through the model.
    If the given number of Monte Carlo Dropout samples is greater than default
    value, MC Dropout is enabled and multiple stochastic passes are performed to
    estimate predictive uncertainty.

    Args:
        model (torch.nn.Module): The PyTorch model to compute forward pass for.
        batch (Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ]): Model batch, either a tuple including the target or inputs
            ready for model.
        device (torch.device): Device on which to run the forward passes.
        num_features (int): Number of features to use.
        num_mc_dropout_samples (int): Number of MC Dropout samples to perform.
        mc_dropout_flag (str): Flag to enable MC Dropout on the model.
        mc_dropout_unbiased_variance (bool): Whether to compute MC Dropout
                                             variance using unbiased estimator.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - outputs_mean: Mean of outputs across one or more MC dropout samples.
            - outputs_var: Variance of outputs across MC dropout samples
                           (None if the number of MC Dropout samples is set to default).

    Raises:
    RuntimeError: If MC Dropout forward pass fails:
        * Forward computation of the model fails due to incompatible inputs or invalid
          tensor shapes (RuntimeError, TypeError).
        * Saving the MC dropout outputs fails because outputs are not valid tensors or
          cannot be unsqueezed (RuntimeError, AttributeError).
        * Concatenation of all MC dropout outputs fails due to mismatched tensor shapes
          or invalid tensor list (RuntimeError, TypeError).
        * Computation of outputs variance fails because the outputs tensor is invalid or
          unsupported operation occurs (RuntimeError, TypeError).
    """
    try:
        # Set model mode depending on the
        # number of samples passed
        if num_mc_dropout_samples > MC_DROPOUT_NUM_SAMPLES_DEFAULT:
            # MC Dropout mode
            set_model_mode(model, MODEL_MC_DROPOUT_MODE, mc_dropout_flag)
        else:
            # Evaluation mode
            set_model_mode(model, MODEL_EVALUATION_MODE)

        all_outputs = []
        with torch.no_grad():
            # For each MC dropout sample provided
            for i in range(num_mc_dropout_samples):
                # Compute forward pass and get the
                # model outputs
                if isinstance(batch, tuple) and len(batch) == num_features + 1:
                    _, outputs = compute_forward(batch, model, device)
                else:
                    outputs = model(*batch)

                # Save the current model outputs
                all_outputs.append(outputs.unsqueeze(0))

        # Concatenate outputs as a tensor
        all_outputs_tensor = torch.cat(all_outputs, dim=0)
        # Calculate outputs mean
        outputs_mean = all_outputs_tensor.mean(dim=0)

        # Calculate outputs variance provided that
        # the number of MC dropout sample is greater
        # than the default value
        outputs_var = None
        if num_mc_dropout_samples > MC_DROPOUT_NUM_SAMPLES_DEFAULT:
            outputs_var = all_outputs_tensor.var(
                dim=0, unbiased=mc_dropout_unbiased_variance
            )

        return outputs_mean, outputs_var
    except (TypeError, AttributeError, RuntimeError) as e:
        msg = "MC Dropout forward failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "batch_len": len(batch) if batch is not None else None,
                "batch_types": (
                    [type(x).__name__ for x in batch]
                    if batch is not None
                    else None
                ),
                "batch_shapes": (
                    [tuple(x.shape) for x in batch]
                    if batch is not None
                    else None
                ),
                "model": type(model).__name__ if model else None,
                "device": str(device),
                "features_num": num_features,
                "mc_dropout_samples_num": num_mc_dropout_samples,
                "mc_dropout_flag": mc_dropout_flag,
                "mc_dropout_unbiased_variance": mc_dropout_unbiased_variance,
                "context": "MC Dropout forward pass",
            },
        )
        raise RuntimeError(msg) from e
