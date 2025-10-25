import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


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
    try:
        debug(
            "Model state dictionary loading started",
            extra={
                "path": path,
                "model_type": type(model).__name__,
                "device": str(device),
                "context": "Model state dictionary loading",
            },
        )

        # Load model state dictionary from specified
        # path, mapping to passed device
        model_state_dict = torch.load(path, map_location=device)

        # Apply state dictionary loaded
        # to provided model
        model.load_state_dict(model_state_dict)

        debug(
            "Model state dictionary loading completed",
            extra={
                "path": path,
                "model_type": type(model).__name__,
                "num_parameters": len(list(model.state_dict().keys())),
                "device": str(device),
                "context": "Model state dictionary loading",
            },
        )

        return model
    except (OSError, RuntimeError, TypeError) as e:
        msg = "Model state dictionary loading failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "path": path,
                "model_type": type(model).__name__,
                "device": str(device),
                "context": "Model state dictionary loading",
            },
        )
        raise RuntimeError(msg) from e
