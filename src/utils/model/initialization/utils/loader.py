import torch
from torch import nn

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def load_model(
    model: nn.Module, device: torch.device, model_path: str
) -> nn.Module:
    """
    Load model weights from a file.

    This function loads the state dictionary from the
    specified path into the given model and maps the
    weights to the provided device.

    Args:
        model (nn.Module): The PyTorch model instance.
        device (torch.device): Device on which to map the model weights.
        model_path (str): Path to the file containing
                          the model state dictionary.

    Returns:
        nn.Module: The model with loaded weights.

    Raises:
        RuntimeError: If an error occurs while loading the model.
    """
    debug(f"Device to load model: {device}")
    debug(f"Path to load model from: {model_path}")

    try:
        # Retrieve model state dictionary from
        # the specified path, using provided
        # device for mapping
        state_dict = torch.load(model_path, map_location=device)

        # Load model state dictionary to model
        model.load_state_dict(state_dict)
    except OSError as e:
        msg = f"Failed to load model from path {model_path}"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Model loaded from {model_path}")

    return model
