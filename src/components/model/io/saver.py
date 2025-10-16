import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def save_model(model: torch.nn.Module, path: str) -> None:
    """
    Save a PyTorch model to the specified path.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): File path to save the model.

    Returns:
        None

    Raises:
        RuntimeError: If saving fails, e.g.:
            * If path to save the model is invalid or inaccessible.
    """
    debug(f"Path to save the model to: {path}")

    try:
        # Save model state dictionary
        # to specified path
        torch.save(model.state_dict(), path)
    except (FileNotFoundError, PermissionError) as e:
        msg = "Failed to save model"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Model saved to {path}")
