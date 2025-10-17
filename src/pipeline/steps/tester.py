from box.box import Box

from components.data_loader.initializer import initialize_data_loader
from components.dataset.access_logs_dataset import AccessLogsDataset
from components.evaluation.model.evaluator import evaluate_model
from components.logs.initializer import initialize_logs, logs_phase
from components.logs.levels.info_logger import info
from components.model.best.initializer import (
    initialize_best_model,
)
from components.model.mode.setter import set_model_mode
from components.visualization.reports.model_evaluation_reporter import (
    generate_model_evaluation_report,
)
from const import (
    DATA_DISTRIBUTION_STATIC_MODE,
    DYNAMIC_MODEL_RESULTS_FILE_NAME,
    EVALUATION_MODEL_MODE,
    LOGS_TESTING_PHASE,
    MODEL_METRICS_CLASS_REPORT_NAME,
    MODEL_METRICS_COHEN_KAPPA_SCORE_NAME,
    MODEL_METRICS_TOP_K_ACCURACY_NAME,
    RESULTS_DIRECTORY_PATH,
    STATIC_MODEL_RESULTS_FILE_NAME,
    TESTING_SPLIT_TYPE,
)
from pipeline.config.configurator import prepare_config


def test_model() -> None:
    """
    Test the trained model.

    This function tests the trained model on the
    testing set. Results are shown via report and
    plots providing model performance insights.

    Returns:
        None
    """
    # Set the new state
    logs_phase.set(LOGS_TESTING_PHASE)

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

    # Setup testing data loader
    _, testing_loader = initialize_data_loader(
        TESTING_SPLIT_TYPE,
        testing_batch_size,
        testing_shuffle,
        AccessLogsDataset,
        config,
    )

    # Trained model setup for testing
    device, criterion, model = initialize_best_model(
        model_params, data_distribution_mode, config, testing_loader
    )

    # Set model in evaluation phase
    set_model_mode(model, EVALUATION_MODEL_MODE)

    # Prepare file name where to save model results
    if data_distribution_mode == DATA_DISTRIBUTION_STATIC_MODE:
        model_results_file_name = STATIC_MODEL_RESULTS_FILE_NAME
    else:
        model_results_file_name = DYNAMIC_MODEL_RESULTS_FILE_NAME

    # Build path to save model metrics
    model_results_save_path = (
        RESULTS_DIRECTORY_PATH
        / data_distribution_mode
        / model_results_file_name
    )

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
        compute_metrics=True,
    )

    # Box for model evaluation metrics
    metrics = Box(metrics)

    # Retrieve metrics
    class_report = getattr(metrics, MODEL_METRICS_CLASS_REPORT_NAME)
    top_k_accuracy = getattr(metrics, MODEL_METRICS_TOP_K_ACCURACY_NAME)
    kappa_score = getattr(metrics, MODEL_METRICS_COHEN_KAPPA_SCORE_NAME)

    # Show report to display testing results
    generate_model_evaluation_report(
        class_report,
        top_k_accuracy,
        kappa_score,
        avg_loss,
        top_k,
    )

    info("Model testing completed")


if __name__ == "__main__":
    test_model()
