"""test_model.py

Module containing the test function for verifying the performance and robustness
of the predictions of the trained model using the Deepchecks library.

This module ensures that the model predictions meets quality constraints defined
in the configuration.

Functions:
    test_model() -> None
        Runs a Deepchecks Suite on the model predictions using the training and
        testing sets based on configuration thresholds.
"""

from deepchecks import Suite
from deepchecks.tabular.checks import (
    SimpleModelComparison,
    TrainTestPerformance,
)

from const import (
    DATA_DYNAMIC_MODE,
    DATA_STATIC_MODE,
)
from tests.const import (
    MODEL_TESTS_ADD_INDEX_COLUMN,
    MODEL_TESTS_DYNAMIC_DATA_RESULTS_SAVE_PATH,
    MODEL_TESTS_REAL_DATA_RESULTS_SAVE_PATH,
    MODEL_TESTS_REMOVE_SEQ_LEN,
    MODEL_TESTS_STATIC_DATA_RESULTS_SAVE_PATH,
    MODEL_TESTS_SUITE_NAME,
)
from tests.helpers.dc_helpers import (
    compute_dc_model_predictions,
    initialize_dc_tests,
    run_dc_suite,
)


def test_model() -> None:
    """Runs the Deepchecks test suite against the trained model.

    This test checks the model predictions by ensuring it adheres to
    predefined configuration thresholds for various checks, including:
        - SimpleModelComparison
        - TrainTestPerformance (Degradation, Imbalance, Min Score)
    The suite is run over the training and testing sets along with the model.

    Returns:
        None
    """
    # ----------------------------
    # Setup
    # ----------------------------
    # Initialization
    (
        training_set,
        _,
        testing_set,
        dc_training_set,
        dc_validation_set,
        dc_testing_set,
        pipeline_config,
        tests_config,
    ) = initialize_dc_tests(
        MODEL_TESTS_ADD_INDEX_COLUMN,
        MODEL_TESTS_REMOVE_SEQ_LEN,
    )

    # ----------------------------
    # Suite building
    # ----------------------------
    # Create a suite of model tests
    suite = Suite(
        MODEL_TESTS_SUITE_NAME,
        SimpleModelComparison(
            strategy=tests_config.model.simple_model_comparison.strategy,
            scorers=tests_config.model.simple_model_comparison.scorers,
            random_state=tests_config.seed.value,
        ).add_condition_gain_greater_than(
            min_allowed_gain=tests_config.model.simple_model_comparison.min_allowed_gain,
        ),
        TrainTestPerformance(
            scorers=tests_config.model.train_test_performance.scorers,
            random_state=tests_config.seed.value,
        ).add_condition_test_performance_greater_than(
            min_score=tests_config.model.train_test_performance.test_min_score,
        ),
        TrainTestPerformance(
            scorers=tests_config.model.train_test_performance.scorers,
            random_state=tests_config.seed.value,
        ).add_condition_train_test_relative_degradation_less_than(
            threshold=tests_config.model.train_test_performance.degradation_threshold,
        ),
    )

    # ----------------------------
    # Inference
    # ----------------------------
    # Get both train and test predictions
    # and probabilities to run DeepChecks tests
    # suite on
    y_pred_train, y_pred_test, y_proba_train, y_proba_test = (
        compute_dc_model_predictions(
            training_set,
            testing_set,
            pipeline_config,
        )
    )

    # ----------------------------
    # Suite running
    # ----------------------------
    # Determine save path based on the
    # data mode
    if pipeline_config.data.general.mode == DATA_STATIC_MODE:
        save_path = MODEL_TESTS_STATIC_DATA_RESULTS_SAVE_PATH
    elif pipeline_config.data.general.mode == DATA_DYNAMIC_MODE:
        save_path = MODEL_TESTS_DYNAMIC_DATA_RESULTS_SAVE_PATH
    else:
        save_path = MODEL_TESTS_REAL_DATA_RESULTS_SAVE_PATH

    # Run the suite over the training and
    # testing sets as well as the model
    run_dc_suite(
        dc_training_set,
        suite,
        str(save_path),
        dc_testing_set=dc_testing_set,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
        y_proba_train=y_proba_train,
        y_proba_test=y_proba_test,
    )


if __name__ == "__main__":
    test_model()
