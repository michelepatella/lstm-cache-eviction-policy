from const import PIPELINE_PHASE_VALIDATION, TRAINING_SPLIT_TYPE
from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.utils.data.AccessLogsDataset import AccessLogsDataset
from lstm_eviction_policy.utils.data.dataloader.data_loader_setup import (
    data_loader_setup,
)
from lstm_eviction_policy.utils.logs.log_utils import phase_var


def validation(config: Config):
    # Set the new pipeline state
    phase_var.set(PIPELINE_PHASE_VALIDATION)

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

    # grid search for best parameters
    best_params = compute_grid_search(training_set, config)

    # set the best parameters and get new config settings
    new_config_settings = save_best_params(best_params, config)

    # print a successful message
    info("✅ Validation successfully completed.")

    return new_config_settings
