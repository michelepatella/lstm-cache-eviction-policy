"""predictions_collector.py

This module provides utility functions for performing batch inference and
aggregating model outputs.

It specifically handles the extraction of predicted class indices from a
PyTorch model using a provided DataLoader and computing device.

Functions:
    collect_model_predictions(
        model: Module,
        dataloader: DataLoader,
        device: torch.device
    ) -> np.ndarray:
        Iterates through a DataLoader to perform inference and return an array of predictions.
"""

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from components.const import TENSOR_CLASS_DIM
from components.device.mover import move_to_device


def collect_model_predictions(
    model: Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Performs inference on a dataset and collects predictions.

    Args:
        model (Module): The PyTorch model used for inference.
        dataloader (DataLoader): The DataLoader containing the input data.
        device (torch.device): The device to run inference on.

    Returns:
        np.ndarray: A numpy array containing the class indices predicted by the model.
    """
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Extract batch components
            x_features, x_keys, _ = batch

            # Move tensors to device
            x_features = move_to_device(x_features, device)
            x_keys = move_to_device(x_keys, device)

            # Make prediction
            outputs = model(x_features, x_keys)

            # Extract and save prediction
            y_hat = torch.argmax(outputs, dim=TENSOR_CLASS_DIM)
            preds.append(y_hat.cpu())
    return torch.cat(preds).numpy()
