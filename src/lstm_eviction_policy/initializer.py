import json
from box import Box
import torch

from utils.model.LSTM import LSTM


def initialize_api():
    with open("model_meta_data.json", "r") as f:
        api_config = json.load(f)
    api_config = Box(api_config)

    model_params = api_config.model.params
    model_path = api_config.model.path

    device = torch.device(api_config.hardware.device_type)

    model = LSTM(model_params, config)

    model.to(device)

    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)
