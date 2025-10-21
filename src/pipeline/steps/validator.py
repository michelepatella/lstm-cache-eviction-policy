import mlflow

from components.data_loader.initializer import initialize_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.dict.operations.merger import merge_dicts
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from components.validation.grid_search.runner import (
    compute_grid_search,
)
from components.yaml.io.saver import save_yaml
from src.const import (
    DATASET_TRAINING_SPLIT_TYPE,
    LOGS_VALIDATION_PHASE, MLFLOW_NESTED_ENABLED,
)
from pipeline.config.configurator import prepare_config
from pipeline.const import CONFIG_FILE_PATH


def validate_model() -> None:
    """
    Validate model to find the best hyperparameters.

    This function validates the model, by orchestrating hyperparameter
    tuning (via time series cross-validation with grid search), and saving
    the best hyperparameters found in a new configuration file.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_VALIDATION_PHASE)
    with mlflow.start_run(run_name=LOGS_VALIDATION_PHASE, nested=MLFLOW_NESTED_ENABLED):

        # Setup
        config = prepare_config()
        initialize_logs()

        info("Validation started")

        # Prepare configuration
        validation_batch_size = config.validation.general.batch_size
        validation_shuffle = config.validation.general.shuffle

        # Load the training set
        dataset_split_type = DATASET_TRAINING_SPLIT_TYPE
        training_set, _ = initialize_data_loader(
            dataset_split_type,
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
        mlflow.log_metrics(
            {
                "num_training_samples": len(training_set),
                "best_avg_loss": best_avg_loss,
            }
        )
        mlflow.log_artifact(CONFIG_FILE_PATH)
        mlflow.end_run()

    info("Validation completed")


if __name__ == "__main__":
    validate_model()
