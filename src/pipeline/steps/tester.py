import logging
import os

import dagshub
import numpy as np
from box import Box
from dotenv import load_dotenv

from components.data_loader.initializer import initialize_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.evaluation.model.evaluator import evaluate_model
from components.logs.handlers.elastic_handler import ElasticHandler
from components.logs.initializer import logs_phase, initialize_logs
from components.logs.levels.info_logger import info
from components.model.best.initializer import (
    initialize_best_model,
)
from pipeline.config.configurator import prepare_config
from pipeline.const import (
    LOGS_TESTING_PHASE,
    MODEL_COMPUTE_METRICS_ENABLED,
    RESULTS_DYNAMIC_MODEL_FILE_PATH,
    RESULTS_STATIC_MODEL_FILE_PATH,
)
from src.const import (
    DAGS_HUB_MLFLOW_ENABLED,
    DAGS_HUB_REPO_NAME_ENV_VAR_NAME,
    DAGS_HUB_REPO_OWNER_ENV_VAR_NAME,
    DATA_DISTRIBUTION_STATIC_MODE,
    DATASET_TESTING_SPLIT_TYPE,
    MLFLOW_NESTED_ENABLED,
)

# Load env variables
load_dotenv()
dabs_hub_repo_owner = os.getenv(DAGS_HUB_REPO_OWNER_ENV_VAR_NAME)
dags_hub_repo_name = os.getenv(DAGS_HUB_REPO_NAME_ENV_VAR_NAME)


def test_model() -> None:
    """Test the trained model.

    This function tests the trained model on the testing set. Results are
    shown via report and plots providing model performance insights.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_TESTING_PHASE)

    dagshub.init(
        repo_owner=dabs_hub_repo_owner,
        repo_name=dags_hub_repo_name,
        mlflow=DAGS_HUB_MLFLOW_ENABLED,
    )

    import mlflow

    with mlflow.start_run(
        run_name=LOGS_TESTING_PHASE,
        nested=MLFLOW_NESTED_ENABLED,
    ):
        # Setup
        config = prepare_config()
        initialize_logs()

        # Prepare configuration
        data_distribution_mode = config.data.mode
        testing_batch_size = config.testing.general.batch_size
        testing_shuffle = config.testing.general.shuffle
        num_features = config.model.general.features
        top_k = config.testing.metrics.top_k
        model_params = config.model.params

        info(
            "Testing started",
            extra={
                "data_distribution_mode": data_distribution_mode,
                "testing_batch_size": testing_batch_size,
                "testing_shuffle": testing_shuffle,
                "features_num": num_features,
                "top_k": top_k,
                "context": "Testing",
            },
        )

        # Setup testing data loader
        testing_set, testing_loader = initialize_data_loader(
            DATASET_TESTING_SPLIT_TYPE,
            testing_batch_size,
            testing_shuffle,
            AccessLogsDataset,
            config,
        )

        # Trained model setup for testing
        device, criterion, model = initialize_best_model(
            model_params,
            data_distribution_mode,
            config,
            testing_loader,
        )

        # Prepare file name where to save model results
        if data_distribution_mode == DATA_DISTRIBUTION_STATIC_MODE:
            model_results_save_path = RESULTS_STATIC_MODEL_FILE_PATH
        else:
            model_results_save_path = RESULTS_DYNAMIC_MODEL_FILE_PATH

        # Evaluate model
        (
            avg_loss,
            metrics,
            all_outputs,
            all_targets,
            _,
        ) = evaluate_model(
            model,
            testing_loader,
            criterion,
            device,
            num_features,
            top_k,
            model_results_save_path,
            compute_metrics=MODEL_COMPUTE_METRICS_ENABLED,
        )

        # Experiment tracking
        metrics = Box(metrics, delimiter="_")
        mlflow.log_params(prepare_config().model_dump())
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
                "top_k_accuracy": metrics.top_k_accuracy,
                "cohen_kappa_score": metrics.cohen_kappa_score,
            },
        )
        mlflow.log_artifact(model_results_save_path)

    info(
        "Testing completed",
        extra={
            "testing_samples_num": len(testing_set),
            "loss_avg": None
            if np.isinf(avg_loss) or np.isnan(avg_loss)
            else float(avg_loss),
            "accuracy": metrics.class_report.accuracy,
            "top_k_accuracy": metrics.top_k_accuracy,
            "model_results_save_path": str(model_results_save_path),
            "context": "Testing",
        },
    )


if __name__ == "__main__":
    test_model()

    # Force logs flush
    for handler in logging.getLogger().handlers:
        if isinstance(handler, ElasticHandler):
            handler.flush_buffer()