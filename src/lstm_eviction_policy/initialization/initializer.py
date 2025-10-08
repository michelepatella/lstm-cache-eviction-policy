from lstm_eviction_policy.initialization.utils.device_selector import (
    select_device,
)
from lstm_eviction_policy.initialization.utils.meta_data_loader import (
    load_meta_data,
)
from lstm_eviction_policy.initialization.utils.model_builder import build_model
from lstm_eviction_policy.initialization.utils.model_state_dict_loader import (
    load_model_state_dict,
)
from lstm_eviction_policy.initialization.utils.model_to_device_mover import (
    move_model_to_device,
)


def initialize_api():

    # Load API configuration encoded
    # as meta data from JSON file
    api_config = load_meta_data("../meta_config.json")

    # Prepare configuration settings
    device_type = api_config.hardware.device_type
    model_params = api_config.model.params
    model_path = api_config.model.path
    min_key = api_config.model.keys.min
    max_key = api_config.model.keys.max
    num_features = api_config.model.num_features
    embedding_dim = api_config.model.embedding_dim

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
