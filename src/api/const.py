"""const.py

Centralized module for defining static constants, network endpoints,
and parameter names related to the API and its underlying microservices.

This module organizes critical infrastructural data and magic strings
to ensure consistency and maintainability across the entire application
architecture, covering inter-service communication, configuration paths,
and standardized naming conventions.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ----------------------------
# API Configuration
# ----------------------------
API_CONFIG_FILE_PATH = PROJECT_ROOT / "src" / "api" / "config" / "config.yaml"


# ----------------------------
# Gateway API
# ----------------------------
GATEWAY_API_ENDPOINT = "/evict"

GATEWAY_API_RETURN_KEYS_TO_EVICT_NAME = "keys_to_evict"
GATEWAY_API_RETURN_API_KWARGS_NAME = "api_kwargs"
GATEWAY_API_RETURN_KEY_SCORES_NAME = "key_scores"
GATEWAY_API_RETURN_PROB_MATRIX_NAME = "prob_matrix"
GATEWAY_API_RETURN_CONF_MATRIX_NAME = "conf_matrix"


# ----------------------------
# Logs
# ----------------------------
LOGS_PHASE_API = "api"


# ----------------------------
# Model
# ----------------------------
MODEL_FILE_PATH = (
    PROJECT_ROOT / "models" / "static" / "trained_static_model.pt"
)


# ----------------------------
# Predictor service
# ----------------------------
PREDICTOR_SERVICE_ENDPOINT = "/predict"
PREDICTOR_SERVICE_ENV_VAR_FULL_BASE_NAME = "API_PREDICTOR_SERVICE_BASE_URL"
PREDICTOR_SERVICE_FULL_URL = (
    os.getenv(PREDICTOR_SERVICE_ENV_VAR_FULL_BASE_NAME, "")
    + PREDICTOR_SERVICE_ENDPOINT
)

PREDICTOR_SERVICE_PARAM_LAST_ACCESSES_NAME = "last_accesses"
PREDICTOR_SERVICE_PARAM_DEVICE_TYPE_NAME = "device_type"
PREDICTOR_SERVICE_PARAM_ROLLOUT_HORIZON_NAME = "rollout_horizon"
PREDICTOR_SERVICE_PARAM_MC_DROPOUT_SAMPLES_NAME = "mc_dropout_samples"
PREDICTOR_SERVICE_PARAM_TIME_STEP_INCREMENT_NAME = "time_step_increment"
PREDICTOR_SERVICE_PARAM_UNBIASED_VARIANCE_NAME = "unbiased_variance"

PREDICTOR_SERVICE_PARAMS = {
    PREDICTOR_SERVICE_PARAM_LAST_ACCESSES_NAME: None,
    PREDICTOR_SERVICE_PARAM_DEVICE_TYPE_NAME: None,
    PREDICTOR_SERVICE_PARAM_ROLLOUT_HORIZON_NAME: None,
    PREDICTOR_SERVICE_PARAM_MC_DROPOUT_SAMPLES_NAME: None,
    PREDICTOR_SERVICE_PARAM_TIME_STEP_INCREMENT_NAME: None,
    PREDICTOR_SERVICE_PARAM_UNBIASED_VARIANCE_NAME: None,
}

PREDICTOR_SERVICE_RETURN_OUTPUTS_NAME = "outputs"
PREDICTOR_SERVICE_RETURN_VARIANCES_NAME = "variances"


# ----------------------------
# Scorer service
# ----------------------------
SCORER_SERVICE_ENDPOINT = "/score"
SCORER_SERVICE_ENV_VAR_BASE_URL_NAME = "API_SCORER_SERVICE_BASE_URL"
SCORER_SERVICE_FULL_URL = (
    os.getenv(SCORER_SERVICE_ENV_VAR_BASE_URL_NAME, "")
    + SCORER_SERVICE_ENDPOINT
)

SCORER_SERVICE_PARAM_OUTPUTS_NAME = "outputs"
SCORER_SERVICE_PARAM_VARIANCES_NAME = "variances"
SCORER_SERVICE_PARAM_CONFIDENCE_LEVEL_NAME = "confidence_level"
SCORER_SERVICE_PARAM_PROB_WEIGHT_NAME = "prob_weight"
SCORER_SERVICE_PARAM_CONF_WEIGHT_NAME = "conf_weight"

SCORER_SERVICE_PARAMS = {
    SCORER_SERVICE_PARAM_OUTPUTS_NAME: None,
    SCORER_SERVICE_PARAM_VARIANCES_NAME: None,
    SCORER_SERVICE_PARAM_CONFIDENCE_LEVEL_NAME: None,
    SCORER_SERVICE_PARAM_PROB_WEIGHT_NAME: None,
    SCORER_SERVICE_PARAM_CONF_WEIGHT_NAME: None,
}

SCORER_SERVICE_RETURN_KEY_SCORES_NAME = "key_scores"
SCORER_SERVICE_RETURN_PROB_MATRIX_NAME = "prob_matrix"
SCORER_SERVICE_RETURN_CONF_MATRIX_NAME = "conf_matrix"


# ----------------------------
# User API Kwargs
# ----------------------------
USER_API_KWARGS_ROLLOUT_HORIZON_NAME = "rollout_horizon"
USER_API_KWARGS_MC_DROPOUT_SAMPLES_NAME = "mc_dropout_samples"
USER_API_KWARGS_CONFIDENCE_LEVEL_NAME = "confidence_level"
USER_API_KWARGS_TIME_STEP_INCREMENT_NAME = "time_step_increment"
USER_API_KWARGS_NUM_EVICTIONS_NAME = "num_evictions"
USER_API_KWARGS_EXCLUDED_KEYS_NAME = "excluded_keys"
USER_API_KWARGS_PROB_WEIGHT_NAME = "prob_weight"
USER_API_KWARGS_CONF_WEIGHT_NAME = "conf_weight"
USER_API_KWARGS_RETURN_ALL_SCORES_NAME = "return_all_scores"
USER_API_KWARGS_RETURN_PROB_CONF_NAME = "return_prob_conf"
USER_API_KWARGS_RETURN_API_KWARGS_NAME = "return_api_kwargs"
