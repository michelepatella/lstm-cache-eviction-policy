import os
import tempfile

import dagshub
from dotenv import load_dotenv

from components.data_loader.builder import build_data_loader
from components.data_loader.initializer import initialize_data_loader
from components.data_loader.targets.extractor import (
    extract_targets_from_data_loader,
)
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.dataset.splits.training_validation_splitter import (
    split_training_validation_sets,
)
from components.logs.initializer import logs_phase
from components.logs.levels.info_logger import info
from components.model.environment.initializer import (
    initialize_model_environment,
)
from components.model.io.locator import get_model_abs_path
from components.model.io.saver import save_model
from components.optimizer.builder import build_optimizer
from components.training.core.epochs_trainer import train_epochs
from pipeline.config.configurator import prepare_config
from src.const import (
    DAGS_HUB_MLFLOW_ENABLED,
    DATASET_TRAINING_SPLIT_TYPE,
    LOGS_TRAINING_PHASE,
    MLFLOW_NESTED_ENABLED,
    MLFLOW_PYTORCH_SAVE_MODEL_PATH, DAGS_HUB_REPO_OWNER_ENV_VAR_NAME, DAGS_HUB_REPO_NAME_ENV_VAR_NAME,
)


# Load env variables
load_dotenv()
dabs_hub_repo_owner = os.getenv(DAGS_HUB_REPO_OWNER_ENV_VAR_NAME)
dags_hub_repo_name = os.getenv(DAGS_HUB_REPO_NAME_ENV_VAR_NAME)


def train_model() -> None:
    """
    Train the model.

    This function trains the model, by orchestrating model and
    training setup, as well as training itself. The best model found
    during training process is saved for further usage.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_TRAINING_PHASE)

    dagshub.init(
        repo_owner=dabs_hub_repo_owner,
        repo_name=dags_hub_repo_name,
        mlflow=DAGS_HUB_MLFLOW_ENABLED,
    )

    import mlflow

    with mlflow.start_run(
        run_name=LOGS_TRAINING_PHASE, nested=MLFLOW_NESTED_ENABLED
    ):

        # Setup
        config = prepare_config()

        info("Training started")

        # Prepare configuration
        data_distribution_mode = config.data.mode
        training_batch_size = config.training.general.batch_size
        training_shuffle = config.training.general.shuffle
        validation_batch_size = config.validation.general.batch_size
        validation_shuffle = config.validation.general.shuffle
        validation_split = config.dataset.split.validation
        model_params = config.model.params
        training_num_epochs = config.training.general.epochs
        optimizer_type = config.training.optimizer.type
        learning_rate = config.training.optimizer.params.learning_rate
        weight_decay = config.training.optimizer.params.weight_decay

        # Get the model path
        model_path = get_model_abs_path(data_distribution_mode)

        # Load the training set and the
        # training loader
        training_set, training_loader = initialize_data_loader(
            DATASET_TRAINING_SPLIT_TYPE,
            training_batch_size,
            validation_shuffle,
            AccessLogsDataset,
            config,
        )

        # Split training set into training
        # and validation sets
        training_set, validation_set = split_training_validation_sets(
            training_set, validation_split
        )

        # Create a loader both for
        # training and validation sets
        training_loader = build_data_loader(
            training_set,
            training_batch_size,
            training_shuffle,
        )
        validation_loader = build_data_loader(
            validation_set,
            validation_batch_size,
            validation_shuffle,
        )

        # Extract targets from training loader
        targets = extract_targets_from_data_loader(training_loader)

        # Model setup for training
        device, criterion, model = initialize_model_environment(
            model_params, config, targets
        )

        # Build optimizer
        optimizer = build_optimizer(
            model, optimizer_type, lr=learning_rate, weight_decay=weight_decay
        )

        # Train the model
        best_avg_loss, model = train_epochs(
            training_num_epochs,
            model,
            training_loader,
            validation_loader,
            optimizer,
            criterion,
            device,
            logs_phase.get(),
            config,
        )

        # Save the best model trained
        save_model(model, model_path)

        # Experiment tracking
        mlflow.log_params(prepare_config().model_dump())
        mlflow.log_metrics(
            {
                "training_samples_num": len(training_set),
                "validation_samples_num": len(validation_set),
                "loss_best_avg": best_avg_loss,
            }
        )
        with tempfile.TemporaryDirectory() as MLFLOW_PYTORCH_SAVE_MODEL_TEMP_PATH:
            mlflow.pytorch.save_model(
                pytorch_model=model,
                path=MLFLOW_PYTORCH_SAVE_MODEL_TEMP_PATH,
            )
            mlflow.log_artifacts(
                local_dir=MLFLOW_PYTORCH_SAVE_MODEL_TEMP_PATH,
                artifact_path=MLFLOW_PYTORCH_SAVE_MODEL_PATH,
            )

    info("Training completed")


if __name__ == "__main__":
    train_model()
