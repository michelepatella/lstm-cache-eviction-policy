from typing import Tuple

from box import Box
from torch import device
from torch.nn import Module

from eviction_policy_api.const import META_DATA_CONFIG_FILE_PATH
from eviction_policy_api.initialization.utils.api_config_extractor import (
    extract_api_config,
)
from eviction_policy_api.initialization.utils.device_selector import (
    select_device,
)
from eviction_policy_api.initialization.utils.meta_data_loader import (
    load_meta_data,
)
from eviction_policy_api.initialization.utils.model_builder import build_model
from eviction_policy_api.initialization.utils.model_state_dict_loader import (
    load_model_state_dict,
)
from eviction_policy_api.initialization.utils.model_to_device_mover import (
    move_model_to_device,
)
from utils.model.LSTM import LSTM


def initialize_api() -> tuple[Module, device, Box]:
    """
    Initialize the eviction policy API.

    This function performs all steps required to
    prepare the eviction policy API:
        - Loads API configuration from JSON file.
        - Extracts API configuration fields.
        - Selects the computation device.
        - Builds the LSTM model instance.
        - Moves the model to the selected device.
        - Loads pre-trained model weights.

    Returns:
        Tuple[Module, torch.device, Box]: Tuple containing the instantiated
                                          LSTM model, the device where the model
                                          resides, and the Box object containing
                                          the API configuration.
    """
    # Load API configuration encoded
    # as meta data from JSON file
    api_config = load_meta_data(META_DATA_CONFIG_FILE_PATH)

    # Extract API configurations from
    # configuration object
    (
        device_type,
        model_params,
        model_path,
        min_key,
        max_key,
        num_features,
        embedding_dim,
    ) = extract_api_config(api_config)

    # Select device according to device type
    device = select_device(device_type)

    # Build model to be used for inference
    model = build_model(
        model_params, min_key, max_key, embedding_dim, num_features
    )

    # Move model built to
    # selected device
    model = move_model_to_device(model, device)

    # Load state dictionary and
    # apply it to the model
    load_model_state_dict(model_path, model, device)

    return model, device, api_config
