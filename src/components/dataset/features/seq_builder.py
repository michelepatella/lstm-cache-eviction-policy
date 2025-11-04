"""seq_builder.py

Module for constructing feature sequences for autoregressive models.

This module provides the `build_feature_seq` function, which:
    - Encodes timestamp arrays trigonometrically into sine and cosine features.
    - Converts key arrays into torch tensors.
    - Adds batch dimensions to both tensors.
    - Moves tensors to the specified device for model input.

Functions:
    build_feature_seq(
        timestamps: np.ndarray,
        keys: np.ndarray,
        device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]
        Builds time and key sequence tensors ready for model consumption.
"""

import numpy as np
import torch

from components.const import (
    TENSOR_FEATURES_DIM,
    TENSOR_OUTPUTS_BATCH_DIM,
    TORCH_DTYPE,
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
    """Build time and key sequence tensors for autoregressive models.

    This function takes an array of timestamps and an array of corresponding
    keys, encodes the timestamps trigonometrically using sine and cosine,
    and stacks them into a tensor with an added batch dimension.
    Similarly, the keys are converted into a tensor with batch dimension.
    Both resulting tensors are then moved to the specified device to be
    ready for model input.

    Args:
        timestamps (np.ndarray): Array of timestamps.
        keys (np.ndarray): Array of keys corresponding to the timestamps.
        device (torch.device): Device where tensors should be allocated.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - times_seq: Tensor with sin/cos features.
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

        # Build times sequence and move to device
        times_seq = torch.tensor(
            np.stack([sin_time, cos_time], axis=TENSOR_FEATURES_DIM),
            dtype=TORCH_DTYPE,
        ).unsqueeze(TENSOR_OUTPUTS_BATCH_DIM)
        times_seq = move_to_device(times_seq, device)

        # Build keys sequence and move to device
        keys_seq = torch.tensor(keys, dtype=TORCH_DTYPE).unsqueeze(
            TENSOR_OUTPUTS_BATCH_DIM,
        )
        keys_seq = move_to_device(keys_seq, device)

        debug(
            "Feature sequences building completed",
            extra={
                "times_seq_shape": times_seq.shape,
                "keys_seq_shape": keys_seq.shape,
                "context": "Feature sequences building",
            },
        )

        return times_seq, keys_seq
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
