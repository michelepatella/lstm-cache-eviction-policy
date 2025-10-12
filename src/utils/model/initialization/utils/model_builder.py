from typing import Dict

from config.classes.ModelConfig import ModelParamsConfig
from utils.model.LSTM import LSTM


def build_model(
    model_params: ModelParamsConfig,
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
        model_params (ModelParamsConfig): Dictionary containing model parameters.
        min_key (int): Minimum key index used in the model.
        max_key (int): Maximum key index used in the model.
        embedding_dim (int): Dimension of the key embedding.
        num_features (int): Number of input features for the model.

    Returns:
        LSTM: Instantiated PyTorch model.
    """
    # Instantiate the model
    model = LSTM(
        model_params, min_key, max_key, embedding_dim, num_features
    )

    return model
