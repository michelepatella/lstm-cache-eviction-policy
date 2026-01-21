"""monitor.py

This module orchestrates the performance monitoring of the production model.

It utilizes the Deepchecks library to run a suite of tests comparing the
performance of the model on new production data versus historical reference
data. It includes checks for baseline comparisons and train-test performance
degradation to ensure the model maintains its predictive quality in production.

Functions:
    monitor_model_performance() -> bool:
        Orchestrates the setup, inference, and execution of Deepchecks performance suites.
"""

from deepchecks import Suite
from deepchecks.tabular.checks import (
    SimpleModelComparison,
    TrainTestPerformance,
)

from components.const import MODEL_PERFORMANCE_MONITORING_RESULTS_FILE_PATH
from components.model.monitoring.initializer import initialize_model_monitoring
from tests.const import MODEL_PERFORMANCE_TESTS_SUITE_NAME
from tests.helpers.dc_helpers import (
    compute_dc_model_predictions,
    create_dc_dataset,
    run_dc_suite,
)


def monitor_model_performance() -> bool:
    """Executes the model performance monitoring suite.

    This method performs the following operational steps:
    1.  Setup: Initializes the monitoring environment, loading datasets,
        configurations, and the production model.
    2.  Dataset preparation: Converts raw dataframes into Deepchecks-compatible
        Dataset objects.
    3.  Preconditions: Validates if the volume of new data meets the minimum
        threshold required for statistically significant testing.
    4.  Suite construction: Builds a Deepchecks Suite with conditions for
        baseline gain, absolute test performance, and relative degradation.
    5.  Inference: Computes predictions and probabilities for both sets.
    6.  Suite execution: Runs the suite and saves the results to a JSON file.

    Returns:
        bool: True if the performance checks pass or if there is insufficient
              data to run (defaulting to safe), False if any performance
              assertion in the suite fails.
    """
    # ----------------------------
    # Setup
    # ----------------------------
    (
        new_df,
        hist_df,
        new_dataset,
        hist_dataset,
        new_dataloader,
        hist_dataloader,
        pipeline_config,
        tests_config,
        model,
        device,
    ) = initialize_model_monitoring()

    # Create a DeepChecks dataset starting
    # from new data collected
    dc_new_dataset = create_dc_dataset(
        new_df.data,
        seq_len=pipeline_config.model.sequence.length,
        index_name=None,
    )
    dc_hist_dataset = create_dc_dataset(
        hist_df.data,
        seq_len=pipeline_config.model.sequence.length,
        index_name=None,
    )

    # ----------------------------
    # Preconditions
    # ----------------------------
    # Check if enough data is available
    if len(new_df) < pipeline_config.training.samples.min:
        return True

    # ----------------------------
    # Suite building
    # ASSUMPTION:
    #   Historical data represents
    #   the training set while new
    #   collected data represents
    #   the testing set.
    # ----------------------------
    # Create a suite of model tests
    suite = Suite(
        MODEL_PERFORMANCE_TESTS_SUITE_NAME,
        SimpleModelComparison(
            strategy=tests_config.model.performance.baseline_comp.strategy,
            scorers=tests_config.model.performance.baseline_comp.scorers,
        ).add_condition_gain_greater_than(
            min_allowed_gain=tests_config.model.performance.baseline_comp.min_gain,
        ),
        TrainTestPerformance(
            scorers=tests_config.model.performance.train_test.scorers,
        ).add_condition_test_performance_greater_than(
            min_score=tests_config.model.performance.train_test.test_min_score,
        ),
        TrainTestPerformance(
            scorers=tests_config.model.performance.train_test.scorers,
        ).add_condition_train_test_relative_degradation_less_than(
            threshold=tests_config.model.performance.train_test.max_degradation,
        ),
    )

    # ----------------------------
    # Inference
    # ----------------------------
    # Get both new and historical predictions
    # and probabilities to run DeepChecks tests
    # suite on
    y_pred_hist, y_pred_new, y_proba_hist, y_proba_new = (
        compute_dc_model_predictions(
            new_dataset,
            hist_dataset,
            pipeline_config,
            training_loader=hist_dataloader,
            testing_loader=new_dataloader,
            model=model,
        )
    )

    # ----------------------------
    # Suite running
    # ----------------------------
    # Run the suite over the training and
    # testing sets as well as the model
    try:
        run_dc_suite(
            dc_hist_dataset,
            suite,
            str(MODEL_PERFORMANCE_MONITORING_RESULTS_FILE_PATH),
            dc_testing_set=dc_new_dataset,
            y_pred_train=y_pred_hist,
            y_pred_test=y_pred_new,
            y_proba_train=y_proba_hist,
            y_proba_test=y_proba_new,
        )
    except AssertionError:
        return False

    return True


if __name__ == "__main__":
    monitor_model_performance()
