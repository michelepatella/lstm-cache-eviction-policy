import torch
from box.box import Box

from const import (
    LOGS_TESTING_PHASE,
    MODEL_METRICS_CLASS_REPORT_NAME,
    MODEL_METRICS_COHEN_KAPPA_SCORE,
    MODEL_METRICS_CONFUSION_MATRIX,
    MODEL_METRICS_TOP_K_ACCURACY,
    TESTING_SPLIT_TYPE,
)
from config.classes.Config import Config
from pipeline.testing.visualization.plots.confusion_matrix_plotter import (
    plot_confusion_matrix,
)
from pipeline.testing.visualization.plots.precision_recall_curve_plotter import (
    plot_precision_recall_curve,
)
from pipeline.testing.visualization.report.model_evaluation_reporter import (
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

    Parameters:
        config (Config): Configuration object.

    Returns:
        None
    """
    # Set the new state
    logs_phase.set(LOGS_TESTING_PHASE)

    # Prepare configuration
    data_distribution_mode = config.data.general.mode
    testing_batch_size = config.testing.batch_size
    testing_shuffle = config.testing.shuffle
    num_keys = config.data.general.keys.max - config.data.general.keys.min + 1
    top_k = config.evaluation.top_k

    # Setup testing data loader
    _, testing_loader = initialize_data_loader(
        TESTING_SPLIT_TYPE,
        testing_batch_size,
        testing_shuffle,
        config,
        AccessLogsDataset,
    )

    # Trained model setup for testing
    device, criterion, model = initialize_trained_model(testing_loader, config)

    # Set model in evaluation phase
    model.eval()

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
        data_distribution_mode,
        compute_metrics=True,
    )

    # Box for model evaluation metrics
    metrics = Box(metrics)

    # Retrieve metrics
    class_report = getattr(metrics, MODEL_METRICS_CLASS_REPORT_NAME)
    top_k_accuracy = getattr(metrics, MODEL_METRICS_TOP_K_ACCURACY)
    kappa_score = getattr(metrics, MODEL_METRICS_COHEN_KAPPA_SCORE)
    confusion_matrix = getattr(metrics, MODEL_METRICS_CONFUSION_MATRIX)

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

    # Show testing-related plots
    plot_precision_recall_curve(
        all_targets, stacked_outputs, num_keys, data_distribution_mode
    )
    plot_confusion_matrix(confusion_matrix, data_distribution_mode)

    info("Model testing completed")
