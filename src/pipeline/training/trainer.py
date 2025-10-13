from config import prepare_config
from const import LOGS_TRAINING_PHASE, TRAINING_SPLIT_TYPE
from pipeline.training.utils.model_saver import save_model
from pipeline.utils.dataset.splitter import split_training_set
from pipeline.utils.training.n_epochs_trainer import train_n_epochs
from utils.data_loader.builder import create_data_loader
from utils.data_loader.initializer import initialize_data_loader
from utils.data_loader.targets_extractor import (
    extract_targets_from_data_loader,
)
from utils.dataset.AccessLogsDataset import AccessLogsDataset
from utils.logs.initializer import logs_phase
from utils.logs.levels.info_logger import info
from utils.model.initialization.model_components_initializer import (
    initialize_model_components,
)
from utils.model.locator import get_model_abs_path


def train_model() -> None:
    """
    Train LSTM model.

    This function trains the LSTM model, by orchestrating
    model and training setup, as well as training itself.
    The best model found during training process is saved
    for further usage.

    Returns:
        None
    """
    # Set the new state
    logs_phase.set(LOGS_TRAINING_PHASE)

    # Read configuration
    config = prepare_config()

    # Prepare configuration
    data_distribution_mode = config.data.generation.mode
    training_batch_size = config.training.general.batch_size
    training_shuffle = config.training.general.shuffle
    validation_batch_size = config.validation.general.batch_size
    validation_shuffle = config.validation.general.shuffle
    model_params = config.model.params
    learning_rate = config.training.optimizer.params.learning_rate
    training_num_epochs = config.training.general.epochs

    # Get the model path
    model_path = get_model_abs_path(data_distribution_mode)

    # Load the training set and the
    # training loader
    training_set, training_loader = initialize_data_loader(
        TRAINING_SPLIT_TYPE,
        training_batch_size,
        validation_shuffle,
        config,
        AccessLogsDataset,
    )

    # Split training set into training
    # and validation sets
    training_set, validation_set = split_training_set(training_set, config)

    # Create a loader both for
    # training and validation sets
    training_loader = create_data_loader(
        training_set,
        training_batch_size,
        training_shuffle,
    )
    validation_loader = create_data_loader(
        validation_set,
        validation_batch_size,
        validation_shuffle,
    )

    # Extract targets from training loader
    targets = extract_targets_from_data_loader(training_loader)

    # Model setup for training
    device, criterion, model, optimizer = initialize_model_components(
        model_params,
        learning_rate,
        config,
        targets
    )

    # Train the model
    _, model = train_n_epochs(
        training_num_epochs,
        model,
        training_loader,
        validation_loader,
        optimizer,
        criterion,
        device,
        config,
    )

    # Save the best model trained
    save_model(model, model_path)

    info("Model training completed")
