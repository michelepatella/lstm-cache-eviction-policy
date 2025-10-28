import numpy as np
from fastapi import FastAPI, HTTPException, status, Body

from components.dataset.features.seq_builder import build_feature_seq
from components.inference.autoregressive_rollout.runner import (
    compute_autoregressive_rollout,
)
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from eviction_policy_api.const import (
    PREDICTOR_SERVICE_ENDPOINT,
    PREDICTOR_SERVICE_RETURN_OUTPUTS_NAME,
    PREDICTOR_SERVICE_RETURN_VARIANCES_NAME,
)
from eviction_policy_api.services.predictor.initializer import (
    initialize_predictor_service,
)

app = FastAPI()


@app.post(PREDICTOR_SERVICE_ENDPOINT)
def predictor_service(
    last_accesses: list[tuple[float, int]] = Body(...),
    model_path: str = Body(...),
    model_params: dict[str, int | float | bool] = Body(...),
    device_type: str = Body(...),
    min_key: int = Body(...),
    max_key: int = Body(...),
    num_features: int = Body(...),
    embedding_dim: int = Body(...),
    rollout_horizon: int = Body(...),
    mc_dropout_samples: int = Body(...),
    time_step_increment: float = Body(...),
) -> dict[str, list[list[float]]]:
    """Predictor microservice for autoregressive rollout.

    This endpoint coordinates the autoregressive rollout service.
    It starts by initializing the predictor model on the specified
    device and constructs feature and keys sequences from the last
    accessed data. The endpoint returns predicted outputs along with
    associated variances coming from autoregressive rollout.

    Args:
        last_accesses (list[tuple[float, int]]): List of tuples containing
                                                 the last accessed keys
                                                 and their corresponding
                                                 timestamps (in hours).
        model_path (str): Path to the pretrained model weights.
        model_params (dict[str, int | float | bool]): Model hyperparameters.
        device_type (str): Device type to run the model on.
        min_key (int): Minimum key value.
        max_key (int): Maximum key value.
        num_features (int): Number of input features for the model.
        embedding_dim (int): Dimension of embedding layer in the model.
        rollout_horizon (int): Number of steps for autoregressive rollout.
        mc_dropout_samples (int): Number of Monte Carlo dropout forward passes
                                  to estimate uncertainty.
        time_step_increment (float): Time increment per rollout step (in hours).

    Returns:
        dict[str, list[list[float]]]: Dictionary containing the outputs of the
                                      autoregressive rollout for each step run,
                                      and their corresponding variances.

    Raises:
        HTTPException: If the predictor service fails:
            * If the last accesses data is malformed or contains invalid types
              (TypeError).
            * If any value in last accesses, model parameters, or device type is
              invalid (ValueError).
            * If a required key is missing in model parameters or last accesses
              (KeyError).
            * If building feature and key sequences fails (RuntimeError).
            * If the autoregressive rollout computation fails (RuntimeError).
    """
    try:
        info(
            "Predictor service started",
            extra={
                "last_accesses_num": len(last_accesses),
                "model_path": str(model_path),
                "model_params": model_params,
                "device_type": device_type,
                "key_min": min_key,
                "key_max": max_key,
                "features_num": num_features,
                "embedding_dim": embedding_dim,
                "rollout_horizon": rollout_horizon,
                "mc_dropout_samples": mc_dropout_samples,
                "time_step_increment": time_step_increment,
                "context": "Predictor service",
            },
        )

        # Setup for autoregressive rollout service
        model, device = initialize_predictor_service(
            model_path,
            model_params,
            device_type,
            min_key,
            max_key,
            num_features,
            embedding_dim,
        )

        # Construct features and keys sequences
        # starting from timestamps and keys extracted
        # from last accesses data
        timestamps = np.array([timestamp for timestamp, _ in last_accesses])
        keys = np.array([key for _, key in last_accesses])
        features_seq, keys_seq = build_feature_seq(
            timestamps,
            keys,
            device,
        )

        # Run autoregressive rollout and get both
        # outputs and variances
        all_outputs, all_variances = compute_autoregressive_rollout(
            model,
            features_seq,
            keys_seq,
            device,
            num_features,
            rollout_horizon,
            mc_dropout_samples,
            time_step_increment,
        )

        info(
            "Predictor service completed",
            extra={
                "outputs_num": len(all_outputs),
                "variances_num": len(all_variances),
                "context": "Predictor service",
            },
        )

        return {
            PREDICTOR_SERVICE_RETURN_OUTPUTS_NAME: [
                o.tolist() for o in all_outputs
            ],
            PREDICTOR_SERVICE_RETURN_VARIANCES_NAME: [
                v.tolist() for v in all_variances
            ],
        }
    except (TypeError, ValueError, KeyError, RuntimeError) as e:
        error(
            "Predictor service failed",
            extra={
                "exception": str(e),
                "last_accesses_num": len(last_accesses),
                "model_path": str(model_path),
                "model_params": model_params,
                "device_type": device_type,
                "key_min": min_key,
                "key_max": max_key,
                "features_num": num_features,
                "embedding_dim": embedding_dim,
                "rollout_horizon": rollout_horizon,
                "mc_dropout_samples": mc_dropout_samples,
                "time_step_increment": time_step_increment,
                "context": "Predictor service",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
