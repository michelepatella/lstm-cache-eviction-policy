import numpy as np
import torch

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
            np.stack([sin_time, cos_time], axis=1),
            dtype=torch.float32,
        ).unsqueeze(0)
        times_seq = move_to_device(times_seq, device)

        # Build keys sequence and move to device
        keys_seq = torch.tensor(keys, dtype=torch.float32).unsqueeze(0)
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
