import torch
from torch.nn import Module

from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def save_model(model: Module, model_save_path: str) -> None:
    """
    Save a PyTorch model to the specified path.

    Args:
        model (Module): The PyTorch model to save.
        model_save_path (str): File path to save the model.

    Returns:
        None

    Raises:
        RuntimeError: If saving fails, e.g.:
            * If path to save the model is invalid or inaccessible.
    """
    debug(f"Path to save the model: {model_save_path}")

    try:
        torch.save(model.state_dict(), model_save_path)
    except (FileNotFoundError, PermissionError) as e:
        msg = "Failed to save model"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Model saved to '{model_save_path}'")
