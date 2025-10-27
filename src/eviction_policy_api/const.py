from pathlib import Path

# ----------------------------
# Project
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ----------------------------
# Configuration
# ----------------------------
API_CONFIG_FILE_PATH = PROJECT_ROOT / "meta_config.json"


# ----------------------------
# Kwargs
# ----------------------------
ROLLOUT_HORIZON_API_KWARG_NAME = "rollout_horizon"
MC_DROPOUT_SAMPLES_API_KWARG_NAME = "mc_dropout_samples"
CONFIDENCE_LEVEL_API_KWARG_NAME = "confidence_level"
TIME_STEP_INCREMENT_API_KWARG_NAME = "time_step_increment"
NUM_EVICTIONS_API_KWARG_NAME = "num_evictions"
EXCLUDE_KEYS_API_KWARG_NAME = "exclude_keys"
PROB_WEIGHT_API_KWARG_NAME = "prob_weight"
CONF_WEIGHT_API_KWARG_NAME = "conf_weight"
TIEBREAK_STRATEGY_API_KWARG_NAME = "tiebreak_strategy"
RETURN_ALL_SCORES_API_KWARG_NAME = "return_all_scores"
RETURN_PROB_CONF_API_KWARG_NAME = "return_prob_conf"

MAX_ROLLOUT_HORIZON = 50
MAX_MC_DROPOUT_SAMPLES = 1000
MAX_TIME_STEP_INCREMENT = 5.0

TIEBREAK_RANDOM_STRATEGY = "random"
TIEBREAK_FIFO_STRATEGY = "fifo"
TIEBREAK_LIFO_STRATEGY = "lifo"
TIEBREAK_STRATEGIES = [
    TIEBREAK_RANDOM_STRATEGY,
    TIEBREAK_FIFO_STRATEGY,
    TIEBREAK_LIFO_STRATEGY,
]

API_KWARGS = {
    ROLLOUT_HORIZON_API_KWARG_NAME,
    MC_DROPOUT_SAMPLES_API_KWARG_NAME,
    CONFIDENCE_LEVEL_API_KWARG_NAME,
    TIME_STEP_INCREMENT_API_KWARG_NAME,
    NUM_EVICTIONS_API_KWARG_NAME,
    EXCLUDE_KEYS_API_KWARG_NAME,
    PROB_WEIGHT_API_KWARG_NAME,
    CONF_WEIGHT_API_KWARG_NAME,
    TIEBREAK_STRATEGY_API_KWARG_NAME,
    RETURN_ALL_SCORES_API_KWARG_NAME,
    RETURN_PROB_CONF_API_KWARG_NAME,
}


# ----------------------------
# API Gateway
# ----------------------------
API_GATEWAY_ENDPOINT = "/evict"


# ----------------------------
# Predictor service
# ----------------------------
PREDICTOR_SERVICE_ENDPOINT = "/predictor"

PREDICTOR_SERVICE_PARAMS = {
    "last_accesses": None,
    "model_path": None,
    "model_params": None,
    "device_type": None,
    "min_key": None,
    "max_key": None,
    "num_features": None,
    "embedding_dim": None,
    "rollout_horizon": None,
    "mc_dropout_samples": None,
    "time_step_increment": None,
}

PREDICTOR_SERVICE_RETURN_OUTPUTS_NAME = "outputs"
PREDICTOR_SERVICE_RETURN_VARIANCES_NAME = "variances"
