from pipeline.config.configurator import prepare_config
from pipeline.const import (
    CONFIG_DIRECTORY_PATH,
    CONFIG_FILE_NAME,
    LOGS_VALIDATION_PHASE,
    TRAINING_SPLIT_TYPE,
)
from pipeline.steps.validation.grid_search.runner import (
    compute_grid_search,
)
from utils.data_loader.building.initializer import initialize_data_loader
from utils.dataset.AccessLogsDataset import AccessLogsDataset
from utils.dict.merger import merge_dicts
from utils.logs.initializer import logs_phase
from utils.logs.levels.info_logger import info
from utils.yaml.saver import save_yaml


def validate_model() -> None:
    """
    Validate model to find the best hyperparameters.

    This function validates the model, by orchestrating
    hyperparameter tuning (via time series cross-validation with
    grid search), and saving the best hyperparameters found
    in a new configuration file.

    Returns:
        None
    """
    # Set the new state
    logs_phase.set(LOGS_VALIDATION_PHASE)

    # Read configuration
    config = prepare_config()

    # Prepare configuration
    validation_batch_size = config.validation.general.batch_size
    validation_shuffle = config.validation.general.shuffle

    # Load the training set
    training_set, _ = initialize_data_loader(
        TRAINING_SPLIT_TYPE,
        validation_batch_size,
        validation_shuffle,
        AccessLogsDataset,
        config,
    )

    # Compute grid search for best parameters
    best_params = compute_grid_search(training_set, config)

    # Merge original dictionary with the best
    # parameters dictionary
    updated_config_dict = merge_dicts(config.model_dump(), best_params)

    # Save updated configuration dictionary
    # as file
    abs_config_path = CONFIG_DIRECTORY_PATH / CONFIG_FILE_NAME
    save_yaml(updated_config_dict, abs_config_path)

    info("Model validation completed")


if __name__ == "__main__":
    validate_model()
