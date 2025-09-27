from const import LOGS_VALIDATION_PHASE, TRAINING_SPLIT_TYPE
from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.utils.data.AccessLogsDataset import AccessLogsDataset
from lstm_eviction_policy.utils.data.dataloader.data_loader_setup import (
    data_loader_setup,
)
from lstm_eviction_policy.utils.logs.levels.info_logger import info
from lstm_eviction_policy.utils.logs.logs_setup import logs_phase
from lstm_eviction_policy.validation.best_params.best_params_saver import (
    save_best_params,
)
from lstm_eviction_policy.validation.tuning.grid_search_optimizer import (
    compute_grid_search,
)


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
    # Set the new pipeline state
    logs_phase.set(LOGS_VALIDATION_PHASE)

    # Prepare configuration
    batch_size = config.validation.general.batch_size
    shuffle = config.validation.general.shuffle

    # Load the training set
    training_set, _ = data_loader_setup(
        TRAINING_SPLIT_TYPE,
        batch_size,
        shuffle,
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
