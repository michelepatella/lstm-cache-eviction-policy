from pathlib import Path

# ----------------------------
# Project & API
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVICTION_POLICY_API_DIRECTORY = PROJECT_ROOT / "eviction_policy_api"


# ----------------------------
# Configuration
# ----------------------------
META_DATA_CONFIG_FILE_NAME = "meta_config.json"
META_DATA_CONFIG_FILE_PATH = PROJECT_ROOT / META_DATA_CONFIG_FILE_NAME


# ----------------------------
# Kwargs
# ----------------------------
ROLLOUT_HORIZON_KWARG_NAME = "rollout_horizon"
MC_DROPOUT_SAMPLES_KWARG_NAME = "mc_dropout_samples"
CONFIDENCE_LEVEL_KWARG_NAME = "confidence_level"
TIME_STEP_INCREMENT_KWARG_NAME = "time_step_increment"
NUM_EVICTIONS_KWARG_NAME = "num_evictions"
EXCLUDE_KEYS_KWARG_NAME = "exclude_keys"
PROB_WEIGHT_KWARG_NAME = "prob_weight"
CONF_WEIGHT_KWARG_NAME = "conf_weight"
TIEBREAK_STRATEGY_KWARG_NAME = "tiebreak_strategy"
RETURN_ALL_SCORES_KWARG_NAME = "return_all_scores"
RETURN_PROB_CONF_KWARG_NAME = "return_prob_conf"

MAX_ROLLOUT_HORIZON = 50
MAX_MC_DROPOUT_SAMPLES = 1000
MAX_TIME_STEP_INCREMENT = 5.0

TIEBREAK_RANDOM = "random"
TIEBREAK_FIFO = "fifo"
TIEBREAK_LIFO = "lifo"
TIEBREAK_STRATEGIES = [
    TIEBREAK_RANDOM,
    TIEBREAK_FIFO,
    TIEBREAK_LIFO,
]

KWARGS = {
    ROLLOUT_HORIZON_KWARG_NAME,
    MC_DROPOUT_SAMPLES_KWARG_NAME,
    CONFIDENCE_LEVEL_KWARG_NAME,
    TIME_STEP_INCREMENT_KWARG_NAME,
    NUM_EVICTIONS_KWARG_NAME,
    EXCLUDE_KEYS_KWARG_NAME,
    PROB_WEIGHT_KWARG_NAME,
    CONF_WEIGHT_KWARG_NAME,
    TIEBREAK_STRATEGY_KWARG_NAME,
    RETURN_ALL_SCORES_KWARG_NAME,
    RETURN_PROB_CONF_KWARG_NAME,
}
