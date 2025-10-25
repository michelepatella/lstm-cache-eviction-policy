from typing import Dict, Optional, Union

from components.model.lstm import LSTM
from pipeline.config.pydantic.config import Config
from pipeline.config.pydantic.sections.model_config import ModelParamsConfig


def build_model(
    model_params: Union[ModelParamsConfig, Dict[str, Union[int, float, bool]]],
    min_key: int,
    max_key: int,
    embedding_dim: int,
    num_features: int,
    config: Optional[Config],
) -> LSTM:
    """
    Build a PyTorch model.

    This function instantiates a PyTorch model with the given parameters
    and configuration.

    Args:
        model_params (Union[ModelParamsConfig, Dict[str, Union[int, float, bool]]]):
            Model parameters.
        min_key (int): Minimum key index used in the model.
        max_key (int): Maximum key index used in the model.
        embedding_dim (int): Dimension of the key embedding for the model.
        num_features (int): Number of input features for the model.
        config (Optional[Config]): Configuration object.

    Returns:
        LSTM: Instantiated PyTorch model.
    """
    # Instantiate the model
    model = LSTM(
        model_params, min_key, max_key, embedding_dim, num_features, config
    )

    return model
