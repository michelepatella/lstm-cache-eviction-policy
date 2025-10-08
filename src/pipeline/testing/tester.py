import torch
from box.box import Box

from config.classes.Config import Config
from const import (
    CONFUSION_MATRIX_FILE_NAME,
    LOGS_TESTING_PHASE,
    MODEL_METRICS_CLASS_REPORT_NAME,
    MODEL_METRICS_COHEN_KAPPA_SCORE_NAME,
    MODEL_METRICS_CONFUSION_MATRIX_NAME,
    MODEL_METRICS_TOP_K_ACCURACY_NAME,
    MODEL_RESULTS_FILE_NAME,
    PLOTS_DIRECTORY_PATH,
    RESULTS_DIRECTORY_PATH,
    TESTING_SPLIT_TYPE,
)
from pipeline.testing.visualization.plotting.confusion_matrix_plotter import (
    plot_confusion_matrix,
)
from pipeline.testing.visualization.reporting.model_evaluation_reporter import (
    generate_model_evaluation_report,
)
from pipeline.utils.evaluation.evaluator import evaluate_model
from utils.data_loader.initializer import initialize_data_loader
from utils.dataset.AccessLogsDataset import AccessLogsDataset
from utils.logs.initializer import logs_phase
from utils.logs.levels.info_logger import info
from utils.model.initialization.trained_model_initializer import (
    initialize_trained_model,
)


def test_model(config: Config) -> None:
    """
    Test the trained model.

    This function tests the trained model on the
    testing set. Results are shown via report and
    plots providing model performance insights.

    Args:
        config (Config): Configuration object.

    Returns:
        None
    """
    # Set the new state
    logs_phase.set(LOGS_TESTING_PHASE)

    # Prepare configuration
    data_distribution_mode = config.data.generation.mode
    testing_batch_size = config.testing.general.batch_size
    testing_shuffle = config.testing.general.shuffle
    num_keys = (
        config.data.generation.keys.max - config.data.generation.keys.min + 1
    )
    top_k = config.testing.metrics.top_k

    # Setup testing data loader
    _, testing_loader = initialize_data_loader(
        TESTING_SPLIT_TYPE,
        testing_batch_size,
        testing_shuffle,
        config,
        AccessLogsDataset,
    )

    # Trained model setup for testing
    device, criterion, model = initialize_trained_model(config, testing_loader)

    # Set model in evaluation phase
    model.eval()

    # Prepare path where to save evaluation metrics
    metrics_save_path = (
        RESULTS_DIRECTORY_PATH
        / data_distribution_mode
        / MODEL_RESULTS_FILE_NAME
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
        config,
        metrics_save_path,
        compute_metrics=True,
    )

    # Box for model evaluation metrics
    metrics = Box(metrics)

    # Retrieve metrics
    class_report = getattr(metrics, MODEL_METRICS_CLASS_REPORT_NAME)
    top_k_accuracy = getattr(metrics, MODEL_METRICS_TOP_K_ACCURACY_NAME)
    kappa_score = getattr(metrics, MODEL_METRICS_COHEN_KAPPA_SCORE_NAME)
    confusion_matrix = getattr(metrics, MODEL_METRICS_CONFUSION_MATRIX_NAME)

    # Show report to display testing results
    generate_model_evaluation_report(
        class_report,
        top_k_accuracy,
        kappa_score,
        avg_loss,
        top_k,
    )

    # Given all the outputs, transform them
    # into an unic tensor with an additional
    # dimension, converting the resulting tensor
    # into a numpy array
    stacked_outputs = torch.stack(all_outputs).numpy()

    # Prepare save paths
    precision_recall_save_path = (
        PLOTS_DIRECTORY_PATH
        / data_distribution_mode
        / PRECISION_RECALL_CURVE_FILE_NAME
    )
    confusion_matrix_save_path = (
        PLOTS_DIRECTORY_PATH
        / data_distribution_mode
        / CONFUSION_MATRIX_FILE_NAME
    )

    # Show testing-related plot
    plot_confusion_matrix(confusion_matrix, confusion_matrix_save_path)

    info("Model testing completed")
