"""const.py

Centralized module for defining static constants, network endpoints,
and parameter names related to the API and its underlying microservices.

This module organizes infrastructural data and magic strings
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
# API
# ----------------------------
API_TITLE = "LSTM Cache Eviction Policy"
API_DESCRIPTION = "Smart cache eviction policy using LSTM predictions."

API_RESPONSE_FIELD_MESSAGE_NAME = "message"
API_RESPONSE_FIELD_STATUS_CODE_NAME = "status-code"
API_RESPONSE_FIELD_METHOD_NAME = "method"
API_RESPONSE_FIELD_TIMESTAMP_NAME = "timestamp"
API_RESPONSE_FIELD_URL_NAME = "url"


# ----------------------------
# API Configuration
# ----------------------------
API_CONFIG_FILE_PATH = PROJECT_ROOT / "src" / "api" / "config" / "config.yaml"


# ----------------------------
# Decider service
# ----------------------------
DECIDER_SERVICE_ENV_VAR_CHANNEL_NAME = "DECIDER_SERVICE_CHANNEL"
DECIDER_SERVICE_CHANNEL = os.getenv(
    DECIDER_SERVICE_ENV_VAR_CHANNEL_NAME,
    "",
)


# ----------------------------
# Featurizer service
# ----------------------------
FEATURIZER_SERVICE_ENV_VAR_CHANNEL_NAME = "FEATURIZER_SERVICE_CHANNEL"
FEATURIZER_SERVICE_CHANNEL = os.getenv(
    FEATURIZER_SERVICE_ENV_VAR_CHANNEL_NAME,
    "",
)


# ----------------------------
# Logs
# ----------------------------
LOGS_PHASE_API = "api"


# ----------------------------
# MLFlow
# ----------------------------
MLFLOW_ENV_VAR_TRACKING_URI_NAME = "MLFLOW_TRACKING_URI"
MLFLOW_TRACKING_URI = os.environ.get(MLFLOW_ENV_VAR_TRACKING_URI_NAME, "")


# ----------------------------
# Predictor service
# ----------------------------
PREDICTOR_SERVICE_ENV_VAR_CHANNEL_NAME = "PREDICTOR_SERVICE_CHANNEL"
PREDICTOR_SERVICE_CHANNEL = os.getenv(
    PREDICTOR_SERVICE_ENV_VAR_CHANNEL_NAME,
    "",
)


# ----------------------------
# Scorer service
# ----------------------------
SCORER_SERVICE_ENV_VAR_CHANNEL_NAME = "SCORER_SERVICE_CHANNEL"
SCORER_SERVICE_CHANNEL = os.getenv(
    SCORER_SERVICE_ENV_VAR_CHANNEL_NAME,
    "",
)
