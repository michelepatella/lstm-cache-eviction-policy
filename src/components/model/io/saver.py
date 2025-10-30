import torch

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def save_model(model: torch.nn.Module, path: str) -> None:
    """Save a PyTorch model.

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
    try:
        debug(
            "Model saving started",
            extra={
                "path": str(path),
                "model_type": type(model).__name__,
                "context": "Model saving",
            },
        )

        # Save model state dictionary
        # at specified path
        torch.save(model, path)

        debug(
            "Model saving completed",
            extra={
                "path": str(path),
                "model_type": type(model).__name__,
                "context": "Model saving",
            },
        )
    except (FileNotFoundError, PermissionError, TypeError, RuntimeError) as e:
        msg = "Model saving failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "path": str(path),
                "model_type": type(model).__name__,
                "context": "Model saving",
            },
        )
        raise RuntimeError(msg) from e
