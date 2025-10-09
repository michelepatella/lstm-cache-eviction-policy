import torch
import pickle
from fastapi import HTTPException, status


def load_model_state_dict(
    path: str, model: torch.nn.Module, device: torch.device
) -> None:
    """
    Load the state dictionary into the given PyTorch model.

    This function loads pre-trained weights from the specified path
    into the provided model, mapping them to the requested device.

    Args:
        path (str): Path to the saved state dictionary (.pt file).
        model (torch.nn.Module): The model instance to load weights into.
        device (torch.device): The device on which to map the weights.

    Returns:
        None

    Raises:
        HTTPException: If an error occurs during model state
                       dictionary loading, e.g.:
            * The file is not found at the specified path.
            * The model state dictionary cannot be loaded due to
              mismatch, corruption, or I/O issues.
    """
    try:
        # Load model state dictionary from specified
        # path, mapping to passed device
        model_state_dict = torch.load(path, map_location=device)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model state dictionary not found at '{path}'",
        ) from e
    except TypeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Type mismatch while loading model state dictionary",
        ) from e
    except pickle.UnpicklingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Corrupted or invalid model state dictionary file",
        ) from e
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"I/O error while loading model state dictionary",
        ) from e

    try:
        # Apply state dictionary loaded
        # to provided model
        model.load_state_dict(model_state_dict)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply state dictionary to model",
        ) from e
    except TypeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Type mismatch while applying state dictionary to model",
        ) from e
