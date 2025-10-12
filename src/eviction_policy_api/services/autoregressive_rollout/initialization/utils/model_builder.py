from typing import Dict

from fastapi import HTTPException, status

from utils.model.LSTM import LSTM


def build_model(
    model_params: Dict[str, int | float | bool],
    min_key: int,
    max_key: int,
    embedding_dim: int,
    num_features: int,
) -> LSTM:
    """
    Instantiate and return a PyTorch model.

    This function instantiates a PyTorch model with
    the given parameters and input configuration, returning it.

    Args:
        model_params (Dict[str, int | float | bool]): Dictionary containing
                                                      model parameters.
        min_key (int): Minimum key index used in the model.
        max_key (int): Maximum key index used in the model.
        embedding_dim (int): Dimension of the key embedding.
        num_features (int): Number of input features for the model.

    Returns:
        LSTM: Instantiated PyTorch model.

    Raises:
        HTTPException: If an error occurs during model instantiation.
    """
    try:
        # Instantiate the model
        model = LSTM(
            model_params, min_key, max_key, embedding_dim, num_features
        )

        return model
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build model",
        ) from e
