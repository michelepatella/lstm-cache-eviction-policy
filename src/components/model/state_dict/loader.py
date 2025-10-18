import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def load_model_state_dict(
    path: str, model: torch.nn.Module, device: torch.device
) -> torch.nn.Module:
    """
    Load the state dictionary into the given PyTorch model.

    This function loads pre-trained weights from the specified path
    into the provided PyTorch model, mapping them to the requested device.

    Args:
        path (str): Path to the saved model state dictionary.
        model (torch.nn.Module): The PyTorch model instance to load weights
                                 into.
        device (torch.device): The device on which to map the weights.

    Returns:
        torch.nn.Module: The model for which state dictionary has been loaded.

    Raises:
        RuntimeError: If loading the model state dictionary fails:
            * Loading the state dictionary from file fails due to file issues,
              invalid path, or incompatible data (OSError, RuntimeError, TypeError).
            * Applying the loaded state dictionary to the model fails due to
              shape mismatch, missing keys, or type issues (RuntimeError, TypeError).
    """
    debug(f"Path to load model state dictionary from: {path}")

    try:
        # Load model state dictionary from specified
        # path, mapping to passed device
        model_state_dict = torch.load(path, map_location=device)
    except (OSError, RuntimeError, TypeError) as e:
        msg = "Failed to load model state dictionary"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    try:
        # Apply state dictionary loaded
        # to provided model
        model.load_state_dict(model_state_dict)
    except (RuntimeError, TypeError) as e:
        msg = "Failed to apply loaded state dictionary to the model"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Model state dictionary loaded from: {path}")

    return model