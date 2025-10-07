from config.classes.Config import Config
from const import LOGS_VALIDATION_PHASE, TRAINING_SPLIT_TYPE
from pipeline.validation.best_params.saver import save_best_params
from pipeline.validation.tuning.grid_search_runner import (
    compute_grid_search,
)
from utils.data_loader.initializer import initialize_data_loader
from utils.dataset.AccessLogsDataset import AccessLogsDataset
from utils.logs.initializer import logs_phase
from utils.logs.levels.info_logger import info


def validate_model(config: Config) -> Config:
    """
    Validate model to find the best hyperparameters.

    This function validates the model, by orchestrating
    hyperparameter tuning (via time series cross-validation with
    grid search), and saving the best hyperparameters found
    in a new configuration.

    Parameters:
        config (Config): Current configuration object.

    Returns:
        Config: Updated configuration object (with the best hyperparameters).
    """
    # Set the new state
    logs_phase.set(LOGS_VALIDATION_PHASE)

    # Prepare configuration
    validation_batch_size = config.validation.general.batch_size
    validation_shuffle = config.validation.general.shuffle

    # Load the training set
    training_set, _ = initialize_data_loader(
        TRAINING_SPLIT_TYPE,
        validation_batch_size,
        validation_shuffle,
        config,
        AccessLogsDataset,
    )

    # Compute grid search for best parameters
    best_params = compute_grid_search(training_set, config)

    # Save the best parameters and
    # get new configuration
    new_config = save_best_params(best_params, config)

    info("Model validation completed")

    return new_config
