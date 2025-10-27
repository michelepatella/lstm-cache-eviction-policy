import torch
from fastapi import HTTPException, status

from components.device.mover import (
    move_to_device,
)
from components.device.selector import (
    select_device,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.model.builder import (
    build_model,
)
from components.model.state_dict.loader import (
    load_model_state_dict,
)


def initialize_predictor_service(
    model_path: str,
    model_params: dict[str, int | float | bool],
    device_type: str,
    min_key: int,
    max_key: int,
    num_features: int,
    embedding_dim: int,
) -> tuple[torch.nn.Module, torch.device]:
    """Initialize the predictor service.

    This function performs all steps required to prepare
    for predictor service:
        - Selects the computation device.
        - Builds the model instance.
        - Moves the model to the selected device.
        - Loads pre-trained model weights.

    Args:
        model_path (str): Path to the saved model weights.
        model_params (dict[str, int | float | bool]): Model hyperparameters.
        device_type (str): Device type to use.
        min_key (int): Minimum key.
        max_key (int): Maximum key.
        num_features (int): Number of input features for the model.
        embedding_dim (int): Dimension of embedding layer.

    Returns:
        tuple[torch.nn.Module, torch.device]:
            - model: The initialized model.
            - device: The computation device on which the model resides.

    Raises:
        HTTPException: If predictor service initialization fails:
            * If device selection fails (RuntimeError).
            * If model building fails (RuntimeError).
            * If moving the model to the device fails (RuntimeError).
            * If loading model weights fails (RuntimeError).
    """
    try:
        debug(
            "Predictor service initialization started",
            extra={
                "model_path": model_path,
                "device_type": device_type,
                "key_min": min_key,
                "key_max": max_key,
                "features_num": num_features,
                "embedding_dim": embedding_dim,
                "context": "Predictor service",
            },
        )

        # Select computation device
        device = select_device(device_type)

        # Build the model
        model = build_model(
            model_params,
            min_key,
            max_key,
            embedding_dim,
            num_features,
            None,
        )

        # Move model to device
        model = move_to_device(model, device)

        # Load pre-trained weights
        load_model_state_dict(model_path, model, device)

        debug(
            "Predictor service initialization completed",
            extra={
                "model_path": model_path,
                "device_type": device_type,
                "key_min": min_key,
                "key_max": max_key,
                "features_num": num_features,
                "embedding_dim": embedding_dim,
                "device": str(device),
                "context": "Predictor service",
            },
        )

        return model, device
    except RuntimeError as e:
        error(
            "Predictor service initialization failed",
            extra={
                "exception": str(e),
                "model_path": model_path,
                "device_type": device_type,
                "key_min": min_key,
                "key_max": max_key,
                "features_num": num_features,
                "embedding_dim": embedding_dim,
                "context": "Predictor service",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
