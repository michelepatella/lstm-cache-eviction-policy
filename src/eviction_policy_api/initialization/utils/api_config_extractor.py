from typing import Tuple, Dict

from box import Box
from fastapi import HTTPException, status


def extract_api_config(
    api_config: Box,
) -> Tuple[str, Dict[str, int | float | bool], str, int, int, int, int]:
    """
    Extract API configuration fields.

    This function extracts and returns API configuration fields
    from the provided API configuration object.

    Args:
        api_config (Box): API configuration object from which
                          to extract fields.

    Returns:
        Tuple[str, Dict[str, int | float | bool], str, int, int, int, int]:
            Tuple containing API configuration fields.

    Raises:
        HTTPException: If an error occurs during API configuration field
                       extraction, e.g.:
            * If a required field is missing from the configuration.
            * If the configuration structure is invalid or malformed.
    """
    try:
        # Extract API configuration fields
        device_type = api_config.hardware.device_type
        model_params = api_config.model.params
        model_path = api_config.model.path
        min_key = api_config.model.keys.min
        max_key = api_config.model.keys.max
        num_features = api_config.model.num_features
        embedding_dim = api_config.model.embedding_dim

        return (
            device_type,
            model_params,
            model_path,
            min_key,
            max_key,
            num_features,
            embedding_dim,
        )
    except AttributeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=("Missing or invalid field in API configuration"),
        ) from e
    except TypeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid configuration structure",
        ) from e
