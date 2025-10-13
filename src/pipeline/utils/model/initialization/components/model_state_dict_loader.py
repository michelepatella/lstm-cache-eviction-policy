import torch

from pipeline.utils.logs.levels.error_logger import error


def load_model_state_dict(
    path: str, model: torch.nn.Module, device: torch.device
) -> torch.nn.Module:
    """
    Load the state dictionary into the given PyTorch model.

    This function loads pre-trained weights from the specified path
    into the provided model, mapping them to the requested device.

    Args:
        path (str): Path to the saved state dictionary (.pt file).
        model (torch.nn.Module): The model instance to load weights into.
        device (torch.device): The device on which to map the weights.

    Returns:
        torch.nn.Module: The model for which state dictionary
                         has been loaded.

    Raises:
        RuntimeError: If an error occurs during model state
                       dictionary loading, e.g.:
            * The file is not found at the specified path.
            * The model state dictionary cannot be loaded due to
              mismatch, corruption, or I/O issues.
    """
    try:
        # Load model state dictionary from specified
        # path, mapping to passed device
        model_state_dict = torch.load(path, map_location=device)

        # Apply state dictionary loaded
        # to provided model
        model.load_state_dict(model_state_dict)

        return model
    except (OSError, RuntimeError, TypeError) as e:
        msg = "Failed to load model state dictionary"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e
