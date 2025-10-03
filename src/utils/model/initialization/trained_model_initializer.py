from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from pipeline.config.classes.Config import Config
from utils.data.data_loader.targets_extractor import (
    extract_targets_from_data_loader,
)
from utils.logs.levels.info_logger import info
from utils.model.initialization.model_components_initializer import (
    initialize_model_components,
)
from utils.model.initialization.utils.model_loader import load_model


def initialize_trained_model(
    data_loader: DataLoader, config: Config
) -> Tuple[torch.device, nn.Module, nn.Module]:
    """
    Prepare a trained model for further usage.

    This function sets up the model components, extracts the target labels
    from the provided data loader, initializes
    the model with the correct device,
    loss function, and loads pre-trained weights.

    Parameters:
        data_loader (DataLoader): DataLoader containing the dataset to be used.
        config (Config): Configuration object.

    Returns:
        Tuple[
        torch.device, nn.Module, nn.Module
        ]: Tuple containing the device on which
           the model is loaded, loss function initialized
           with class weights, and pre-trained model.
    """
    # Prepare configuration
    model_params = config.model.params
    learning_rate = config.training.optimizer.params.learning_rate
    model_path = config.model.general.path

    # Extract targets from
    # provided data loader
    targets = extract_targets_from_data_loader(data_loader)

    # Setup for model components
    device, criterion, model, _ = initialize_model_components(
        model_params, learning_rate, targets, config
    )

    # Load the trained model
    model = load_model(model, device, model_path)

    info("Trained model initialization completed")

    return device, criterion, model
