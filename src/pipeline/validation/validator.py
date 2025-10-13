from config import prepare_config
from config.io.update.updater import update_config
from const import LOGS_VALIDATION_PHASE, TRAINING_SPLIT_TYPE
from pipeline.validation.best_params.saver import save_best_params
from pipeline.validation.tuning.grid_search_runner import (
    compute_grid_search,
)
from utils.data_loader.initializer import initialize_data_loader
from utils.dataset.AccessLogsDataset import AccessLogsDataset
from utils.logs.initializer import logs_phase
from utils.logs.levels.info_logger import info


def validate_model() -> None:
    """
    Validate model to find the best hyperparameters.

    This function validates the model, by orchestrating
    hyperparameter tuning (via time series cross-validation with
    grid search), and saving the best hyperparameters found
    in a new configuration.

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
        config,
        AccessLogsDataset,
    )

    # Compute grid search for best parameters
    best_params = compute_grid_search(training_set, config)

    # Save the best parameters and
    # get new configuration
    new_config = save_best_params(best_params, config)

    # Update configuration file and
    # get the updated object
    new_config = update_config(new_config.model_dump(), prepare_config)

    info("Model validation completed")
