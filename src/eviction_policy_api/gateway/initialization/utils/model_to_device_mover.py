import torch
from fastapi import HTTPException, status


def move_model_to_device(
    model: torch.nn.Module, device: torch.device
) -> torch.nn.Module:
    """
    Move a PyTorch model to the specified device.

    This function moves the provided model to the requested device
    and returns it.

    Args:
        model (torch.nn.Module): The PyTorch model to move.
        device (torch.device): The target device for computation.

    Returns:
        torch.nn.Module: The model on the specified device.

    Raises:
        HTTPException: If an error occurs while moving the model
                       to the specified device, e.g.:
            * If the device type is invalid or not recognized.
            * If the model cannot be moved due to runtime constraints.
    """
    try:
        # Move model to device
        model = model.to(device)

        return model
    except TypeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid device '{device}' to move model to",
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to move model to device '{device}'",
        ) from e
