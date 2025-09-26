from const import PIPELINE_PHASE_VALIDATION, TRAINING_SPLIT_TYPE
from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.utils.data.AccessLogsDataset import AccessLogsDataset
from lstm_eviction_policy.utils.data.dataloader.data_loader_setup import (
    data_loader_setup,
)
from lstm_eviction_policy.utils.logs.log_utils import info, phase_var
from lstm_eviction_policy.validation.tuning.grid_search_optimizer import (
    compute_grid_search,
)


def validate_model(config: Config):
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

    # Compute grid search for best parameters
    best_params = compute_grid_search(training_set, config)

    # Set the best parameters and
    # get new config settings
    new_config = save_best_params(best_params, config)

    info("Validation completed")

    return new_config
