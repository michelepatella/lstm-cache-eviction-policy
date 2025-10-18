import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info


def save_model(model: torch.nn.Module, path: str) -> None:
    """
    Save a PyTorch model.

    This function saves a PyTorch model to the specified path.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        path (str): File path to save the model.

    Returns:
        None

    Raises:
        RuntimeError: If saving the model fails:
            * File not found or path invalid (FileNotFoundError).
            * Permission denied for the specified path (PermissionError).
            * Model contains non-serializable objects (TypeError).
            * General runtime failure during saving (RuntimeError).
    """
    debug(f"Path to save the model at: {path}")

    try:
        # Save model state dictionary
        # at specified path
        torch.save(model.state_dict(), path)
    except (FileNotFoundError, PermissionError, TypeError, RuntimeError) as e:
        msg = "Failed to save model"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Model saved to: {path}")
