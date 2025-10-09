from pathlib import Path

# ----------------------------
# Project
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVICTION_POLICY_API_DIRECTORY = PROJECT_ROOT / "eviction_policy_api"


# ----------------------------
# Configuration
# ----------------------------
META_DATA_CONFIG_FILE_NAME = "meta_config.json"
META_DATA_CONFIG_FILE_PATH = PROJECT_ROOT / META_DATA_CONFIG_FILE_NAME
