"""validator.py

Pipeline step module responsible for model validation and hyperparameter
optimization using Grid Search and Time Series Cross-Validation.

This module provides the `validate_model` function, which orchestrates the
loading of the training dataset, generates all parameter combinations,
runs the cross-validation and search process to find the optimal set of
hyperparameters, and updates the project configuration file with the best
parameters found.

Functions:
    validate_model() -> None
        Executes the hyperparameter tuning workflow and saves the updated
        configuration.
"""

import logging

import dagshub
import mlflow
import numpy as np
from box import Box

from components.data_loader.initializer import initialize_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.dict.operations.merger import merge_dicts
from components.logs.handlers.grafana_loki_handler import GrafanaLokiHandler
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from components.ray.initializer import initialize_ray
from components.seed.setter import set_seed
from components.validation.grid_search.runner import (
    compute_grid_search,
)
from components.validation.search_space.combinator import (
    get_parameters_combination,
)
from components.yaml.io.saver import save_yaml
from const import (
    DATASET_TRAINING_SPLIT_TYPE,
    LOGS_LOGGER_NAME,
    LOGS_PHASE_VALIDATION,
    MLFLOW_NESTED,
)
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import (
    PIPELINE_CONFIG_FILE_PATH,
    DAGS_HUB_DVC,
    DAGS_HUB_REPO_NAME,
    DAGS_HUB_REPO_OWNER,
)


def validate_model() -> None:
    """Validate model to find the best hyperparameters.

    This function validates the model, by orchestrating hyperparameter
    tuning (via time series cross-validation with grid search), and saving
    the best hyperparameters found in a new configuration file.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_PHASE_VALIDATION)

    dagshub.init(
        repo_owner=DAGS_HUB_REPO_OWNER,
        repo_name=DAGS_HUB_REPO_NAME,
        dvc=DAGS_HUB_DVC,
    )

    with mlflow.start_run(
        run_name=LOGS_PHASE_VALIDATION,
        nested=MLFLOW_NESTED,
    ):
        # Setup
        pipeline_config = prepare_pipeline_config()
        initialize_logs(
            logging.getLevelName(pipeline_config.logs.level),
            GrafanaLokiHandler(),
        )
        initialize_ray(
            pipeline_config.resources.general.num_cpus,
            pipeline_config.resources.general.num_gpus,
        )

        # Prepare configuration
        validation_batch_size = (
            pipeline_config.data_loader.batch_size.validation
        )
        validation_shuffle = pipeline_config.data_loader.shuffle.validation
        seed = pipeline_config.seed.value

        # Ensure reproducibility
        set_seed(seed)

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
            pipeline_config,
        )

        # Get all parameter combinations
        params_combinations = get_parameters_combination(pipeline_config)

        # Compute grid search for best parameters
        best_params_dict, best_avg_loss = compute_grid_search(
            training_set,
            params_combinations,
            pipeline_config,
        )
        best_params_dict = Box(best_params_dict)

        # Prepare the best parameters to be saved
        best_params = Box(pipeline_config.model_dump())
        best_params.model.params.hidden_size = best_params_dict.hidden_size
        best_params.model.params.num_layers = best_params_dict.num_layers
        best_params.model.params.dropout = best_params_dict.dropout
        best_params.optimizer.params.learning_rate = (
            best_params_dict.learning_rate
        )

        # Merge original dictionary with the best
        # parameters dictionary
        updated_config_dict = merge_dicts(
            pipeline_config.model_dump(), best_params
        )

        # Save updated configuration dictionary as file
        save_yaml(
            Box(updated_config_dict).to_dict(), PIPELINE_CONFIG_FILE_PATH
        )

        # Experiment tracking
        mlflow.log_params(prepare_pipeline_config().model_dump())
        mlflow.log_metrics(
            {
                "training_samples_num": len(training_set),
                "loss_best_avg": None
                if np.isinf(best_avg_loss) or np.isnan(best_avg_loss)
                else float(best_avg_loss),
            },
        )
        mlflow.log_artifact(PIPELINE_CONFIG_FILE_PATH)

    info(
        "Validation completed",
        extra={
            "training_samples_num": len(training_set),
            "loss_avg_best": None
            if np.isinf(best_avg_loss) or np.isnan(best_avg_loss)
            else float(best_avg_loss),
            "config_save_path": str(PIPELINE_CONFIG_FILE_PATH),
            "context": "Validation",
        },
    )


if __name__ == "__main__":
    validate_model()

    # Force logs flush
    for handler in logging.getLogger(LOGS_LOGGER_NAME).handlers:
        if isinstance(handler, GrafanaLokiHandler):
            handler.flush_buffer_sync()
