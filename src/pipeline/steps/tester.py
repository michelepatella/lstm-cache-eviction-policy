import mlflow
from box import Box

from components.data_loader.initializer import initialize_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.evaluation.model.evaluator import evaluate_model
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from components.model.best.initializer import (
    initialize_best_model,
)
from const import (
    DATA_DISTRIBUTION_STATIC_MODE,
    DATASET_TESTING_SPLIT_TYPE,
)
from pipeline.config.configurator import prepare_config
from pipeline.const import (
    LOGS_TESTING_PHASE,
    MODEL_COMPUTE_METRICS_ENABLED,
    RESULTS_DYNAMIC_MODEL_FILE_PATH,
    RESULTS_STATIC_MODEL_FILE_PATH,
)


def test_model() -> None:
    """
    Test the trained model.

    This function tests the trained model on the testing set. Results are
    shown via report and plots providing model performance insights.

    Returns:
        None
    """
    # Set the new pipeline step
    logs_phase.set(LOGS_TESTING_PHASE)
    mlflow.start_run(run_name=LOGS_TESTING_PHASE, nested=True)

    # Setup
    config = prepare_config()
    initialize_logs()

    info("Testing started")

    # Prepare configuration
    data_distribution_mode = config.data.mode
    testing_batch_size = config.testing.general.batch_size
    testing_shuffle = config.testing.general.shuffle
    num_features = config.model.general.features
    top_k = config.testing.metrics.top_k
    model_params = config.model.params

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
        model_params, data_distribution_mode, config, testing_loader
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

    mlflow.log_metrics(
        {
            "num_testing_samples": len(testing_set),
            "avg_loss": avg_loss,
            "class_report": Box(metrics).class_report,
            "top_k_accuracy": Box(metrics).top_k_accuracy,
            "cohen_kappa_score": Box(metrics).cohen_kappa_score,
        }
    )
    mlflow.log_artifact(model_results_save_path)
    mlflow.end_run()

    info("Testing completed")


if __name__ == "__main__":
    test_model()
