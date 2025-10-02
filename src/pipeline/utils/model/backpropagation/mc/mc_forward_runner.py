from typing import Tuple, Union

import torch
from torch import nn

from const import MC_DROPOUT_SAMPLES_DEFAULT
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.info_logger import info
from pipeline.utils.model.backpropagation.forward_runner import compute_forward
from pipeline.utils.model.backpropagation.mc.mc_dropout_activator import (
    enable_mc_dropout,
)


def mc_forward_passes(
    model: nn.Module,
    inputs: Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ],
    device: torch.device,
    num_features: int,
    mc_dropout_samples: int = MC_DROPOUT_SAMPLES_DEFAULT,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """
    Perform forward passes with optional Monte Carlo (MC) Dropout.

    This function executes one or more forward passes through the model.
    If the given number of Monte Carlo dropout samples is greater than default
    value, MC Dropout is enabled and multiple
    stochastic passes are performed to
    estimate predictive uncertainty.

    Parameters:
        model (nn.Module): The PyTorch model to evaluate.
        inputs (Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ]): Model inputs.
        device (torch.device): Device on which to run the forward passes.
        num_features (int): Number of features to use.
        mc_dropout_samples (int): Number of MC Dropout samples to perform.

    Returns:
        Tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor
        ]: Tuple containing a tensor for mean
           of outputs across one or more MC
           samples, a tensor for variance of
           outputs across one or more MC
           (computed if and only if the provided
           number of Monte Carlo dropout samples
           is greater than default value), and a
           tensor of all MC outputs concatenated
           along a new dimension.
    """
    # Enable MC Dropout if more than one MC
    # dropout sample
    if mc_dropout_samples > MC_DROPOUT_SAMPLES_DEFAULT:
        # Enable MC dropout during inference
        # (i.e., set the model to training mode)
        enable_mc_dropout(model)

        debug(f"MC Dropout enabled for {mc_dropout_samples} forward passes")
    else:
        # Set the model to evaluation mode
        # during inference
        model.eval()

        debug("Model set to evaluation mode (no MC Dropout)")

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

    # Calculate outputs mean
    outputs_mean = outputs_mc_tensor.mean(dim=0)

    # Calculate outputs variance provided that
    # the number of MC dropout sample is greater
    # than the default value
    if mc_dropout_samples > MC_DROPOUT_SAMPLES_DEFAULT:
        outputs_var = outputs_mc_tensor.var(dim=0, unbiased=False)
    else:
        outputs_var = None

    debug(f"Outputs tensor shape: {outputs_mc_tensor.shape}")
    debug(f"Outputs mean shape: {outputs_mean.shape}")
    if outputs_var is not None:
        debug(f"Outputs variance shape: {outputs_var.shape}")

    info("MC forward pass(es) completed")

    return outputs_mean, outputs_var, outputs_mc_tensor
