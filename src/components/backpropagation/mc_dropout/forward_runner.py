from typing import Optional, Tuple, Union

import torch

from components.backpropagation.core.forward_runner import compute_forward
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from components.model.mode.setter import set_model_mode
from const import (
    EVALUATION_MODEL_MODE,
    MC_DROPOUT_FLAG,
    MC_DROPOUT_MODEL_MODE,
    MC_DROPOUT_NUM_SAMPLES_DEFAULT,
    MC_DROPOUT_UNBIASED_VARIANCE,
)


def compute_mc_dropout_forward(
    model: torch.nn.Module,
    batch: Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ],
    device: torch.device,
    num_features: int,
    num_mc_dropout_samples: Optional[int] = MC_DROPOUT_NUM_SAMPLES_DEFAULT,
    mc_dropout_flag: Optional[str] = MC_DROPOUT_FLAG,
    mc_dropout_unbiased_variance: Optional[
        bool
    ] = MC_DROPOUT_UNBIASED_VARIANCE,
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
        num_mc_dropout_samples (Optional[int]): Number of MC Dropout samples
                                                to perform.
        mc_dropout_flag (Optional[str]): Flag to enable MC Dropout on the model.
        mc_dropout_unbiased_variance (Optional[bool]): Whether to compute MC Dropout
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
            set_model_mode(model, MC_DROPOUT_MODEL_MODE, mc_dropout_flag)
        else:
            # Evaluation mode
            set_model_mode(model, EVALUATION_MODEL_MODE)

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

                debug(
                    f"MC sample {i + 1}/{num_mc_dropout_samples} "
                    f"completed, outputs shape: {outputs.shape}"
                )

        # Concatenate outputs as a tensor
        all_outputs_tensor = torch.cat(all_outputs, dim=0)
        debug(f"Outputs tensor shape: {all_outputs_tensor.shape}")

        # Calculate outputs mean
        outputs_mean = all_outputs_tensor.mean(dim=0)
        debug(f"Outputs mean shape: {outputs_mean.shape}")

        # Calculate outputs variance provided that
        # the number of MC dropout sample is greater
        # than the default value
        outputs_var = None
        if num_mc_dropout_samples > MC_DROPOUT_NUM_SAMPLES_DEFAULT:
            outputs_var = all_outputs_tensor.var(
                dim=0, unbiased=mc_dropout_unbiased_variance
            )
            debug(f"Outputs variance shape: {outputs_var.shape}")

        info("MC forward pass(es) completed")

        return outputs_mean, outputs_var
    except (TypeError, AttributeError, RuntimeError) as e:
        msg = "Failed to compute MC Dropout forward"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
