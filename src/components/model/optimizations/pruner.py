import torch
from torch.nn.utils import prune

from components.const import (
    MODEL_OPTIMIZATION_PRUNING_PARAMS,
    MODEL_OPTIMIZATION_PRUNING_TARGET_MODULES,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def prune_model(
    model: torch.nn.Module,
    amount: float,
    target_modules: tuple[
        type[torch.nn.Module], ...,
    ] = MODEL_OPTIMIZATION_PRUNING_TARGET_MODULES,
    params: tuple[str] = MODEL_OPTIMIZATION_PRUNING_PARAMS,
) -> torch.nn.Module:
    """Apply pruning to a PyTorch model.

    This function applies pruning to the specified modules
    of a PyTorch model, according to the given parameters and amount.

    Args:
        model (torch.nn.Module): The PyTorch model to prune.
        amount (float): Fraction of parameters to prune (0–1).
        target_modules (tuple[type[torch.nn.Module], ...]):
            Layer types to apply pruning on.
        params (tuple[str]): Names of parameters to prune.

    Returns:
        torch.nn.Module: The pruned model.

    Raises:
        RuntimeError: If model pruning fails:
            * Layer does not have the specified parameter (AttributeError).
            * Invalid amount or pruning configuration (ValueError).
            * The module or parameter is incompatible with pruning
              (TypeError).
    """
    try:
        debug(
            "Model pruning started",
            extra={
                "model_type": type(model).__name__,
                "amount": amount,
                "target_modules": [m.__name__ for m in target_modules],
                "params": params,
                "context": "Model pruning",
            },
        )

        # Prune each requested model parameter
        # at any target module
        for name, module in model.named_modules():
            if isinstance(module, target_modules):
                for pname, _ in list(module.named_parameters()):
                    if any(p in pname for p in params):
                        # Apply pruning according to
                        # the specified amount
                        prune.L1Unstructured.apply(
                            module, name=pname, amount=amount,
                        )
                        prune.remove(module, pname)

        debug(
            "Model pruning completed",
            extra={
                "model_type": type(model).__name__,
                "amount": amount,
                "target_modules": [m.__name__ for m in target_modules],
                "params": params,
                "context": "Model pruning",
            },
        )

        return model
    except (AttributeError, ValueError, TypeError) as e:
        msg = "Model pruning failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "model_type": type(model).__name__,
                "amount": amount,
                "target_modules": [m.__name__ for m in target_modules],
                "params": params,
                "context": "Model pruning",
            },
        )
        raise RuntimeError(msg) from e
