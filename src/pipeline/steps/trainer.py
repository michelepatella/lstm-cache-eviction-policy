"""trainer.py

Pipeline step module responsible for setting up the environment, executing
the training process, and saving the resulting best model.

This module provides the `train_model` function, which orchestrates the
initialization of the model, optimizer, data loaders, and the training
loop over multiple epochs. It handles the splitting of the training set
into training and validation sets, applies optimizations (pruning and
quantization), and saves the final trained model.

Functions:
    train_model() -> None
        Executes the full model training and optimization workflow,
        saving the final, optimized model.
"""

import logging
import tempfile

import dagshub
import mlflow
import numpy as np

from components.const import DATASET_PROCESSED_FEATURE_COLUMNS
from components.data_loader.builder import build_data_loader
from components.data_loader.initializer import initialize_data_loader
from components.data_loader.targets.extractor import (
    extract_targets_from_data_loader,
)
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.dataset.splits.training_validation_splitter import (
    split_training_validation_sets,
)
from components.logs.handlers.grafana_loki_handler import GrafanaLokiHandler
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from components.model.builder import build_model
from components.model.environment.initializer import (
    initialize_model_environment,
)
from components.model.io.locator import get_model_abs_path
from components.model.io.saver import save_model
from components.model.optimizations.pruner import prune_model
from components.model.optimizations.quantizer import quantize_model
from components.optimizer.builder import build_optimizer
from components.seed.setter import set_seed
from components.training.core.epochs_trainer import train_epochs
from const import (
    DATASET_TRAINING_SPLIT_TYPE,
    LOGS_LOGGER_NAME,
    MLFLOW_NESTED,
)
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import (
    DAGS_HUB_DVC,
    DAGS_HUB_REPO_NAME,
    DAGS_HUB_REPO_OWNER,
    DATASET_PROCESSED_TYPE,
    LOGS_PHASE_TRAINING,
    MLFLOW_ARTIFACT_PATH,
)


def train_model() -> None:
    """Train the model.

    This function trains the model, by orchestrating model and
    training setup, as well as training itself. The best model found
    during training process is saved for further usage.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_PHASE_TRAINING)

    dagshub.init(
        repo_owner=DAGS_HUB_REPO_OWNER,
        repo_name=DAGS_HUB_REPO_NAME,
        dvc=DAGS_HUB_DVC,
    )

    with mlflow.start_run(
        run_name=LOGS_PHASE_TRAINING,
        nested=MLFLOW_NESTED,
    ):
        # Setup
        pipeline_config = prepare_pipeline_config()
        initialize_logs(
            logging.getLevelName(pipeline_config.logs.level),
            GrafanaLokiHandler(),
        )

        # Prepare configuration
        data_mode = pipeline_config.data.general.mode
        min_key = pipeline_config.data.general.keys.min
        max_key = pipeline_config.data.general.keys.max
        training_batch_size = pipeline_config.data_loader.batch_size.training
        training_shuffle = pipeline_config.data_loader.shuffle.training
        training_device = pipeline_config.resources.devices.training
        validation_batch_size = (
            pipeline_config.data_loader.batch_size.validation
        )
        validation_shuffle = pipeline_config.data_loader.shuffle.validation
        validation_split = pipeline_config.dataset.splits.validation
        model_params = pipeline_config.model.params
        embedding_dim = pipeline_config.model.sequence.embedding.dimension
        training_num_epochs = pipeline_config.training.epochs
        optimizer_type = pipeline_config.optimizer.type
        learning_rate = pipeline_config.optimizer.params.learning_rate
        weight_decay = pipeline_config.optimizer.params.weight_decay
        pruning_amount = pipeline_config.model.optimizations.pruning.amount
        quantization_dtype = (
            pipeline_config.model.optimizations.quantization.dtype
        )
        quantization_engine = (
            pipeline_config.model.optimizations.quantization.engine
        )
        num_features = len(DATASET_PROCESSED_FEATURE_COLUMNS)
        seed = pipeline_config.seed.value

        # Ensure reproducibility
        set_seed(seed)

        info(
            "Training started",
            extra={
                "data_mode": data_mode,
                "training_batch_size": training_batch_size,
                "training_shuffle": training_shuffle,
                "validation_batch_size": validation_batch_size,
                "validation_shuffle": validation_shuffle,
                "epochs": training_num_epochs,
                "optimizer": optimizer_type,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "context": "Training",
            },
        )

        # Get the model path
        model_path = get_model_abs_path(data_mode)

        # Load the training set and the
        # training loader
        training_set, training_loader = initialize_data_loader(
            DATASET_PROCESSED_TYPE,
            DATASET_TRAINING_SPLIT_TYPE,
            training_batch_size,
            validation_shuffle,
            AccessLogsDataset,
            pipeline_config,
        )

        # Split training set into training
        # and validation sets
        training_set, validation_set = split_training_validation_sets(
            training_set,
            validation_split,
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
            targets,
            training_device,
            pipeline_config,
            model_params=model_params,
        )

        # Build optimizer
        optimizer = build_optimizer(
            model,
            optimizer_type,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Train the model
        best_avg_loss, best_model_weights, num_epochs_run = train_epochs(
            training_num_epochs,
            model,
            training_loader,
            validation_loader,
            optimizer,
            criterion,
            device,
            logs_phase.get(),
            pipeline_config,
        )

        # Re-built model with best weights found
        model = build_model(
            model_params.__dict__,
            min_key,
            max_key,
            embedding_dim,
            num_features,
            pipeline_config,
        )
        model.load_state_dict(best_model_weights)

        # Optimize trained model before saving
        # it, applying pruning and quantization before
        model = prune_model(model, pruning_amount)
        model = quantize_model(model, quantization_dtype, quantization_engine)

        # Save the best model trained
        save_model(model, model_path)

        # Experiment tracking
        mlflow.log_params(prepare_pipeline_config().model_dump())
        mlflow.log_metrics(
            {
                "training_samples_num": len(training_set),
                "validation_samples_num": len(validation_set),
                "loss_best_avg": None
                if np.isinf(best_avg_loss) or np.isnan(best_avg_loss)
                else float(best_avg_loss),
                "epochs_run_num": num_epochs_run,
            },
        )
        with (
            tempfile.TemporaryDirectory() as MLFLOW_PYTORCH_SAVE_MODEL_TEMP_PATH
        ):
            mlflow.pytorch.save_model(
                pytorch_model=model,
                path=MLFLOW_PYTORCH_SAVE_MODEL_TEMP_PATH,
            )
            mlflow.log_artifacts(
                local_dir=MLFLOW_PYTORCH_SAVE_MODEL_TEMP_PATH,
                artifact_path=MLFLOW_ARTIFACT_PATH,
            )

    info(
        "Training completed",
        extra={
            "training_samples_num": len(training_set),
            "validation_samples_num": len(validation_set),
            "loss_best_avg": float(best_avg_loss)
            if not (np.isinf(best_avg_loss) or np.isnan(best_avg_loss))
            else None,
            "epochs_run_num": num_epochs_run,
            "model_save_path": str(model_path),
            "context": "Training",
        },
    )


if __name__ == "__main__":
    train_model()

    # Force logs flush
    for handler in logging.getLogger(LOGS_LOGGER_NAME).handlers:
        if isinstance(handler, GrafanaLokiHandler):
            handler.flush_buffer_sync()
