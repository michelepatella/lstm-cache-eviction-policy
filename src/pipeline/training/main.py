from const import LOGS_TRAINING_PHASE, TRAINING_SPLIT_TYPE
from pipeline.config.classes.Config import Config
from pipeline.training.utils.model_saver import save_model
from pipeline.utils.data.AccessLogsDataset import AccessLogsDataset
from pipeline.utils.data.dataloader.data_loader_builder import (
    create_data_loader,
)
from pipeline.utils.data.dataloader.data_loader_setup import (
    data_loader_setup,
)
from pipeline.utils.data.dataloader.data_loader_targets_extractor import (
    extract_targets_from_data_loader,
)
from pipeline.utils.data.dataset.dataset_splitter import (
    split_training_set,
)
from pipeline.utils.logs.levels.info_logger import info
from pipeline.utils.logs.logs_setup import logs_phase
from pipeline.utils.model.setup.model_training_setup import (
    model_training_setup,
)
from pipeline.utils.training.n_epochs_trainer import train_n_epochs


def train_model(config: Config) -> None:
    """
    Train LSTM model.

    This function trains the LSTM model, by orchestrating
    model and training setup, as well as training itself.
    The best model found during training process is saved
    for further usage.

    Parameters:
        config (Config): Configuration object.

    Returns:
        None
    """
    # Set the new pipeline state
    logs_phase.set(LOGS_TRAINING_PHASE)

    # Prepare configuration
    training_batch_size = config.training.general.batch_size
    training_shuffle = config.training.general.shuffle
    validation_batch_size = config.validation.general.batch_size
    validation_shuffle = config.validation.general.shuffle
    model_params = config.model.params
    learning_rate = config.training.optimizer.params.learning_rate
    training_num_epochs = config.training.general.epochs.count
    model_save_path = config.model.general.path

    # Load the training set and the
    # training loader
    training_set, training_loader = data_loader_setup(
        TRAINING_SPLIT_TYPE,
        training_batch_size,
        training_shuffle,
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
    device, criterion, model, optimizer = model_training_setup(
        model_params,
        learning_rate,
        targets,
        config,
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
    save_model(model, model_save_path)

    info("Model training completed")
