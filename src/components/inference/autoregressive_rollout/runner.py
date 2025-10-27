import numpy as np
import torch

from components.backpropagation.mc_dropout.forward_runner import (
    compute_mc_dropout_forward,
)
from components.const import (
    AUTOREGRESSIVE_ROLLOUT_LAST_TIME_BATCH_IDX,
    AUTOREGRESSIVE_ROLLOUT_LAST_TIME_IDX,
    DATASET_COS_TIME_COLUMN_IDX,
    DATASET_SIN_TIME_COLUMN_IDX,
    DATASET_TARGET_COLUMN_IDX,
    TENSOR_OUTPUTS_BATCH_DIM,
    TENSOR_TEMPORAL_DIM,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from components.time.transforms.trig_decoder import (
    decode_time_trigonometrically,
)
from components.time.transforms.trig_encoder import (
    encode_time_trigonometrically,
)


def compute_autoregressive_rollout(
    model: torch.nn.Module,
    features_seq: torch.Tensor,
    keys_seq: torch.Tensor,
    device: torch.device,
    num_features: int,
    rollout_horizon: int,
    mc_dropout_samples: int,
    time_step_increment: float,
    sin_time_idx: int = DATASET_SIN_TIME_COLUMN_IDX,
    cos_time_idx: int = DATASET_COS_TIME_COLUMN_IDX,
    target_idx: int = DATASET_TARGET_COLUMN_IDX,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Perform autoregressive rollout.

    This function executes autoregressive prediction for a
    specified horizon. At each step:
        - Performs MC Dropout forward.
        - Extracts predicted keys.
        - Updates feature and key sequences with predicted values.
        - Advances time using trigonometrical encoding.

    Args:
        model (torch.nn.Module): PyTorch model to perform rollout with.
        features_seq (torch.Tensor): Input features sequence.
        keys_seq (torch.Tensor): Input key sequence.
        device (torch.device): Device for computation.
        num_features (int): Number of features in the model input.
        rollout_horizon (int): Number of autoregressive steps.
        mc_dropout_samples (int): Number of MC dropout samples.
        time_step_increment (float): Increment per rollout step.
        sin_time_idx (int): Column index for sin time feature.
        cos_time_idx (int): Column index for cos time feature.
        target_idx (int): Column index representing target (key)
                          in outputs.

    Returns:
        tuple[list[torch.Tensor], list[torch.Tensor]]:
            - all_outputs: List of model outputs at each step.
            - all_variances: List of variances for each step.

    Raises:
        RuntimeError: If rollout computation fails:
            * If tensor concatenation fails due to shape mismatch (RuntimeError).
            * If tensor indexing fails (IndexError).
            * If tensor or numpy operations receive incompatible types (TypeError).
    """
    try:
        debug(
            "Autoregressive rollout started",
            extra={
                "model": type(model).__name__,
                "features_seq_shape": features_seq.shape,
                "keys_seq_shape": keys_seq.shape,
                "rollout_horizon": rollout_horizon,
                "mc_dropout_samples": mc_dropout_samples,
                "device": str(device),
                "context": "Autoregressive rollout",
            },
        )

        # Extract cos and sin time as features from the
        # first batch and the last request
        last_time = decode_time_trigonometrically(
            features_seq[
                AUTOREGRESSIVE_ROLLOUT_LAST_TIME_BATCH_IDX,
                AUTOREGRESSIVE_ROLLOUT_LAST_TIME_IDX,
                sin_time_idx,
            ].item(),
            features_seq[
                AUTOREGRESSIVE_ROLLOUT_LAST_TIME_BATCH_IDX,
                AUTOREGRESSIVE_ROLLOUT_LAST_TIME_IDX,
                cos_time_idx,
            ].item(),
        )

        # Perform autoregressive rollout
        # horizon times, collecting outputs
        # and variances at each step
        all_outputs, all_variances = [], []
        for _ in range(rollout_horizon):
            # Use the current batch to perform
            # MC dropout and get mean and variance
            # of outputs
            batch = (features_seq, keys_seq)
            outputs_mean, outputs_variance = compute_mc_dropout_forward(
                model,
                batch,
                device,
                num_features,
                mc_dropout_samples,
            )

            # Save outputs and variances
            all_outputs.append(
                outputs_mean.squeeze(dim=TENSOR_OUTPUTS_BATCH_DIM),
            )
            all_variances.append(
                outputs_variance.squeeze(dim=TENSOR_OUTPUTS_BATCH_DIM),
            )

            # Update the sequence of keys by appending
            # the predicted one at the current step
            pred_key = outputs_mean.argmax(dim=target_idx).unsqueeze(1)
            keys_seq = torch.cat([keys_seq[:, 1:], pred_key], dim=target_idx)

            # Calculate new sin and cos time obtained by adding
            # a time step increment simulating passing of time,
            # resulting in new features for the next rollout step
            new_sin_time, new_cos_time = encode_time_trigonometrically(
                np.array([last_time + time_step_increment]),
            )
            new_features = torch.tensor(
                [[new_sin_time[0], new_cos_time[0]]],
                device=device,
            ).unsqueeze(0)

            # Update the sequence of features by appending
            # the new features
            features_seq = torch.cat(
                [features_seq[:, 1:, :], new_features],
                dim=TENSOR_TEMPORAL_DIM,
            )

        debug(
            "Autoregressive rollout completed",
            extra={
                "outputs_num": len(all_outputs),
                "variances_num": len(all_variances),
                "features_seq_shape": features_seq.shape,
                "keys_seq_shape": keys_seq.shape,
                "context": "Autoregressive rollout",
            },
        )

        return all_outputs, all_variances
    except (RuntimeError, IndexError, TypeError) as e:
        msg = "Autoregressive rollout failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "model": model.__class__.__name__,
                "features_seq_shape": getattr(features_seq, "shape", None),
                "keys_seq_shape": getattr(keys_seq, "shape", None),
                "rollout_horizon": rollout_horizon,
                "mc_dropout_samples": mc_dropout_samples,
                "device": str(device),
                "context": "Autoregressive rollout",
            },
        )
        raise RuntimeError(msg) from e
