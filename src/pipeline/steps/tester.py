"""tester.py

Pipeline step module responsible for the final evaluation of the
trained model on the dedicated testing set.

This module provides the `test_model` function, which orchestrates the
loading of the best-trained model, initializes the testing data loader,
runs the final evaluation, and computes various performance metrics.

Functions:
    test_model() -> None
        Initializes the model, evaluates its final performance on the
        testing dataset, and logs all results.
"""

import logging

import dagshub
import mlflow
import numpy as np
from box import Box

from components.data_loader.initializer import initialize_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.evaluation.model.evaluator import evaluate_model
from components.logs.handlers.grafana_loki_handler import GrafanaLokiHandler
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from components.model.best.initializer import (
    initialize_best_model,
)
from components.seed.setter import set_seed
from pipeline.config.configurator import prepare_pipeline_config
from pipeline.const import (
    DAGS_HUB_DVC,
    DAGS_HUB_REPO_NAME,
    DAGS_HUB_REPO_OWNER,
    DATASET_PROCESSED_TYPE,
    LOGS_PHASE_TESTING,
    MODEL_COMPUTE_METRICS_TESTING,
)
from src.const import (
    DATASET_TESTING_SPLIT_TYPE,
    LOGS_LOGGER_NAME,
    MLFLOW_NESTED,
)


def test_model() -> None:
    """Test the trained model.

    This function tests the trained model on the testing set. Results are
    shown via report and plots providing model performance insights.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_PHASE_TESTING)

    dagshub.init(
        repo_owner=DAGS_HUB_REPO_OWNER,
        repo_name=DAGS_HUB_REPO_NAME,
        dvc=DAGS_HUB_DVC,
    )

    with mlflow.start_run(
        run_name=LOGS_PHASE_TESTING,
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
        testing_batch_size = pipeline_config.data_loader.testing.batch_size
        testing_shuffle = pipeline_config.data_loader.testing.shuffle
        testing_device = pipeline_config.resources.devices.testing
        qengine = pipeline_config.model.optimizations.quantization.engine
        num_workers = max(
            pipeline_config.resources.general.num_cpus,
            pipeline_config.resources.general.num_gpus,
        )
        seed = pipeline_config.seed.value

        # Ensure reproducibility
        set_seed(seed)

        info(
            "Testing started",
            extra={
                "data_mode": data_mode,
                "testing_batch_size": testing_batch_size,
                "testing_shuffle": testing_shuffle,
                "workers_num": num_workers,
                "context": "Testing",
            },
        )

        # Setup testing data loader
        testing_set, testing_loader = initialize_data_loader(
            DATASET_PROCESSED_TYPE,
            DATASET_TESTING_SPLIT_TYPE,
            testing_batch_size,
            testing_shuffle,
            AccessLogsDataset,
            pipeline_config,
        )

        # Trained model setup for testing
        device, criterion, model = initialize_best_model(
            data_mode,
            testing_device,
            pipeline_config,
            testing_loader,
            qengine=qengine,
        )

        # Evaluate model
        (avg_loss, metrics, *_) = evaluate_model(
            model,
            testing_loader,
            criterion,
            device,
            num_workers,
            compute_metrics=MODEL_COMPUTE_METRICS_TESTING,
        )
        # Experiment tracking
        metrics = Box(metrics, delimiter="_")
        mlflow.log_params(prepare_pipeline_config().model_dump())
        mlflow.log_metrics(
            {
                "testing_samples_num": len(testing_set),
                "loss_avg": None
                if np.isinf(avg_loss) or np.isnan(avg_loss)
                else float(avg_loss),
                "accuracy": metrics.class_report.accuracy,
                "macro_precision": metrics.class_report.macro_avg.precision,
                "macro_recall": metrics.class_report.macro_avg.recall,
                "macro_f1": metrics.class_report.macro_avg.f1_score,
                "macro_support": metrics.class_report.macro_avg.support,
                "weighted_precision": metrics.class_report.weighted_avg.precision,
                "weighted_recall": metrics.class_report.weighted_avg.recall,
                "weighted_f1": metrics.class_report.weighted_avg.f1_score,
                "weighted_support": metrics.class_report.weighted_avg.support,
                "cohen_kappa_score": metrics.cohen_kappa_score,
            },
        )

    info(
        "Testing completed",
        extra={
            "testing_samples_num": len(testing_set),
            "loss_avg": None
            if np.isinf(avg_loss) or np.isnan(avg_loss)
            else float(avg_loss),
            "accuracy": metrics.class_report.accuracy,
            "context": "Testing",
        },
    )


if __name__ == "__main__":
    test_model()

    # Force logs flush
    for handler in logging.getLogger(LOGS_LOGGER_NAME).handlers:
        if isinstance(handler, GrafanaLokiHandler):
            handler.flush_buffer_sync()
