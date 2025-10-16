from typing import Tuple, Union

import torch

from components.backpropagation.core.forward_runner import compute_forward
from components.logs.levels.debug_logger import debug
from components.logs.levels.info_logger import info
from components.model.mode.setter import set_model_mode
from const import (
    EVALUATION_MODEL_MODE,
    MC_DROPOUT_FLAG,
    MC_DROPOUT_MODEL_MODE,
    MC_DROPOUT_NUM_SAMPLES_DEFAULT,
)


def compute_mc_dropout_forward(
    model: torch.nn.Module,
    inputs: Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ],
    device: torch.device,
    num_features: int,
    mc_dropout_samples: int = MC_DROPOUT_NUM_SAMPLES_DEFAULT,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    Perform forward passes with optional Monte Carlo (MC) Dropout.

    This function executes one or more forward passes through the model.
    If the given number of Monte Carlo dropout samples is greater than default
    value, MC Dropout is enabled and multiple
    stochastic passes are performed to
    estimate predictive uncertainty.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        inputs (Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ]): Model inputs.
        device (torch.device): Device on which to run the forward passes.
        num_features (int): Number of features to use.
        mc_dropout_samples (int): Number of MC Dropout samples to perform.

    Returns:
        Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
            - outputs_mean: Mean of outputs across one or more MC dropout samples.
            - outputs_var: Variance of outputs across MC dropout samples (None if only default single pass).
            - outputs_mc_tensor: All MC outputs concatenated along a new dimension.
    """
    # Set model mode depending on the
    # number of samples passed
    if mc_dropout_samples > MC_DROPOUT_NUM_SAMPLES_DEFAULT:
        # MC Dropout mode
        set_model_mode(model, MC_DROPOUT_MODEL_MODE, MC_DROPOUT_FLAG)
    else:
        # Evaluation mode
        set_model_mode(model, EVALUATION_MODEL_MODE)

    # Initialization of MC outputs
    outputs_mc = []

    with torch.no_grad():
        # For each MC dropout sample provided
        for i in range(mc_dropout_samples):
            # Compute forward pass and get the
            # model outputs
            if isinstance(inputs, tuple) and len(inputs) == num_features + 1:
                _, outputs = compute_forward(inputs, model, None, device)
            else:
                outputs = model(*inputs)

            # Save the current model outputs
            outputs_mc.append(outputs.unsqueeze(0))

            debug(
                f"MC sample {i + 1}/{mc_dropout_samples} "
                f"completed, outputs shape: {outputs.shape}"
            )

    # Aggregate outputs
    outputs_mc_tensor = torch.cat(outputs_mc, dim=0)
    debug(f"Outputs tensor shape: {outputs_mc_tensor.shape}")

    # Calculate outputs mean
    outputs_mean = outputs_mc_tensor.mean(dim=0)
    debug(f"Outputs mean shape: {outputs_mean.shape}")

    # Calculate outputs variance provided that
    # the number of MC dropout sample is greater
    # than the default value
    outputs_var = None
    if mc_dropout_samples > MC_DROPOUT_NUM_SAMPLES_DEFAULT:
        outputs_var = outputs_mc_tensor.var(dim=0, unbiased=False)
        debug(f"Outputs variance shape: {outputs_var.shape}")

    info("MC forward pass(es) completed")

    return outputs_mean, outputs_var, outputs_mc_tensor
