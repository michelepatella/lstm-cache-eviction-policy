import logging
import os

import dagshub
import numpy as np
from dotenv import load_dotenv

from components.data_loader.initializer import initialize_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.dict.operations.merger import merge_dicts
from components.logs.handlers.elastic_handler import ElasticHandler
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from components.validation.grid_search.runner import (
    compute_grid_search,
)
from components.yaml.io.saver import save_yaml
from pipeline.config.configurator import prepare_config
from pipeline.const import CONFIG_FILE_PATH
from src.const import (
    DAGS_HUB_MLFLOW_ENABLED,
    DAGS_HUB_REPO_NAME_ENV_VAR_NAME,
    DAGS_HUB_REPO_OWNER_ENV_VAR_NAME,
    DATASET_TRAINING_SPLIT_TYPE,
    LOGS_VALIDATION_PHASE,
    MLFLOW_NESTED_ENABLED,
)

# Load env variables
load_dotenv()
dabs_hub_repo_owner = os.getenv(DAGS_HUB_REPO_OWNER_ENV_VAR_NAME)
dags_hub_repo_name = os.getenv(DAGS_HUB_REPO_NAME_ENV_VAR_NAME)


def validate_model() -> None:
    """Validate model to find the best hyperparameters.

    This function validates the model, by orchestrating hyperparameter
    tuning (via time series cross-validation with grid search), and saving
    the best hyperparameters found in a new configuration file.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_VALIDATION_PHASE)

    dagshub.init(
        repo_owner=dabs_hub_repo_owner,
        repo_name=dags_hub_repo_name,
        mlflow=DAGS_HUB_MLFLOW_ENABLED,
    )

    import mlflow

    with mlflow.start_run(
        run_name=LOGS_VALIDATION_PHASE,
        nested=MLFLOW_NESTED_ENABLED,
    ):
        # Setup
        config = prepare_config()
        initialize_logs()

        # Prepare configuration
        validation_batch_size = config.validation.general.batch_size
        validation_shuffle = config.validation.general.shuffle

        info(
            "Validation started",
            extra={
                "batch_size": validation_batch_size,
                "shuffle": validation_shuffle,
                "context": "Validation",
            },
        )

        # Load the training set
        training_set, _ = initialize_data_loader(
            DATASET_TRAINING_SPLIT_TYPE,
            validation_batch_size,
            validation_shuffle,
            AccessLogsDataset,
            config,
        )

        # Compute grid search for best parameters
        best_params, best_avg_loss = compute_grid_search(training_set, config)

        # Merge original dictionary with the best
        # parameters dictionary
        updated_config_dict = merge_dicts(config.model_dump(), best_params)

        # Save updated configuration dictionary as file
        save_yaml(updated_config_dict, CONFIG_FILE_PATH)

        # Experiment tracking
        mlflow.log_params(prepare_config().model_dump())
        mlflow.log_metrics(
            {
                "training_samples_num": len(training_set),
                "loss_best_avg": None
                if np.isinf(best_avg_loss) or np.isnan(best_avg_loss)
                else float(best_avg_loss),
            },
        )
        mlflow.log_artifact(CONFIG_FILE_PATH)

    info(
        "Validation completed",
        extra={
            "training_samples_num": len(training_set),
            "loss_avg_best": None
            if np.isinf(best_avg_loss) or np.isnan(best_avg_loss)
            else float(best_avg_loss),
            "config_save_path": str(CONFIG_FILE_PATH),
            "context": "Validation",
        },
    )


if __name__ == "__main__":
    validate_model()

    # Force logs flush
    for handler in logging.getLogger().handlers:
        if isinstance(handler, ElasticHandler):
            handler.flush_buffer()
