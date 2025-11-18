"""seq_builder.py

Module for constructing features and key sequences for autoregressive models.

This module provides the `build_feature_seq` function, which:
    - Encodes timestamp arrays trigonometrically into sine and cosine features.
    - Computes local frequency and local recency features for each key
      within the given sequence window.
    - Stacks all features into a single tensor suitable for model input.
    - Converts key arrays into torch tensors.
    - Adds batch dimensions to both feature and key tensors.
    - Moves tensors to the specified device for model consumption.

Functions:
    build_feature_seq(
        timestamps: np.ndarray,
        keys: np.ndarray,
        device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]
        Builds features and key sequence tensors ready for model consumption.
"""

import numpy as np
import torch

from components.const import (
    TENSOR_BATCH_DIM,
    TENSOR_FEATURES_DIM,
    TORCH_DTYPE_FEATURES,
    TORCH_DTYPE_TARGET,
)
from components.dataset.features.local.local_frequencies_calculator import (
    calculate_local_frequencies,
)
from components.dataset.features.local.local_recencies_calculator import (
    calculate_local_recencies,
)
from components.device.mover import move_to_device
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.time.transforms.trig_encoder import (
    encode_time_trigonometrically,
)


def build_feature_seq(
    timestamps: np.ndarray,
    keys: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build features and key sequence tensors for autoregressive models.

    This function takes an array of timestamps and an array of corresponding
    keys, encodes the timestamps trigonometrically using sine and cosine, as
    well as calculates local frequencies and recencies for keys, and stacks
    them into a tensor with an added batch dimension. Similarly, the keys are
    converted into a tensor with batch dimension. Both resulting tensors are
    then moved to the specified device to be ready for model input.

    Args:
        timestamps (np.ndarray): Array of timestamps.
        keys (np.ndarray): Array of keys corresponding to the timestamps.
        device (torch.device): Device where tensors should be allocated.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - features_seq: Tensor with features.
            - keys_seq: Tensor with keys.

    Raises:
        RuntimeError:
        * If inputs are not np.ndarray (TypeError).
        * If tensor creation or device movement fails (RuntimeError).
    """
    try:
        debug(
            "Feature sequences building started",
            extra={
                "timestamps_shape": getattr(timestamps, "shape", None),
                "keys_shape": getattr(keys, "shape", None),
                "device": str(device),
                "context": "Feature sequences building",
            },
        )

        # Encode time trigonometrically
        sin_time, cos_time = encode_time_trigonometrically(timestamps)

        # Compute local frequencies and recencies
        # for the keys
        seq_len = len(keys)
        local_frequencies = calculate_local_frequencies(keys.tolist(), seq_len)
        local_recencies = calculate_local_recencies(keys.tolist(), seq_len)

        # Stack all features
        features = np.stack(
            [sin_time, cos_time, local_frequencies, local_recencies],
            axis=TENSOR_FEATURES_DIM,
        )

        # Convert to tensor and move to device
        features_seq = torch.tensor(
            features,
            dtype=TORCH_DTYPE_FEATURES,
        ).unsqueeze(
            TENSOR_BATCH_DIM,
        )
        features_seq = move_to_device(features_seq, device)

        # Build keys sequence and move to device
        keys_seq = torch.tensor(keys, dtype=TORCH_DTYPE_TARGET).unsqueeze(
            TENSOR_BATCH_DIM,
        )
        keys_seq = move_to_device(keys_seq, device)

        debug(
            "Feature sequences building completed",
            extra={
                "features_seq_shape": features_seq.shape,
                "keys_seq_shape": keys_seq.shape,
                "context": "Feature sequences building",
            },
        )

        return features_seq, keys_seq
    except (TypeError, RuntimeError) as e:
        msg = "Feature sequences building failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "timestamps_shape": getattr(timestamps, "shape", None),
                "keys_shape": getattr(keys, "shape", None),
                "device": str(device),
                "context": "Feature sequences building",
            },
        )
        raise RuntimeError(msg) from e
