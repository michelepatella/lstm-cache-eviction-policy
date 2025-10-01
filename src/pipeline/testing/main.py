import torch
from box import Box
from sklearn.metrics import cohen_kappa_score

from const import (
    LOGS_TESTING_PHASE,
    MODEL_METRICS_CLASS_REPORT_NAME,
    MODEL_METRICS_COHEN_KAPPA_SCORE,
    MODEL_METRICS_CONFUSION_MATRIX,
    MODEL_METRICS_TOP_K_ACCURACY,
    TESTING_SPLIT_TYPE,
)
from pipeline.config.classes.Config import Config
from pipeline.testing.visualization.confusion_matrix_plotter import (
    plot_confusion_matrix,
)
from pipeline.testing.visualization.model_evaluation_reporter import (
    generate_model_evaluation_report,
)
from pipeline.testing.visualization.precision_recall_curve_plotter import (
    plot_precision_recall_curve,
)
from pipeline.utils.data.AccessLogsDataset import AccessLogsDataset
from pipeline.utils.data.dataloader.data_loader_setup import (
    data_loader_setup,
)
from pipeline.utils.logs.levels.info_logger import info
from pipeline.utils.logs.logs_setup import logs_phase
from pipeline.utils.model.evaluation.model_evaluator import (
    evaluate_model,
)
from pipeline.utils.model.setup.trained_model_setup import (
    trained_model_setup,
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
    # Set the new pipeline state
    logs_phase.set(LOGS_TESTING_PHASE)

    # Prepare configuration
    testing_batch_size = config.testing.batch_size
    testing_shuffle = config.testing.shuffle
    num_keys = config.data.general.keys.max - config.data.general.keys.min + 1
    top_k = config.evaluation.top_k

    # Setup testing data loader
    _, testing_loader = data_loader_setup(
        TESTING_SPLIT_TYPE,
        testing_batch_size,
        testing_shuffle,
        config,
        AccessLogsDataset,
    )

    # Trained model setup for testing
    device, criterion, model = trained_model_setup(testing_loader, config)

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
        all_targets,
        stacked_outputs,
        num_keys,
    )
    plot_confusion_matrix(confusion_matrix)

    info("Model testing completed")
