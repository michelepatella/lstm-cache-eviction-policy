"""model_tests_config.py

Module defining the Pydantic models for configuring tests related to
model performance, robustness, and behavior.

These classes enforce validation rules for thresholds on metrics like AUC,
inference time, performance degradation, and bias checks.

Classes:
    ModelSimpleModelComparisonTestsConfig(BaseModel): Configuration for the simple model
                                                      comparison check.
    ModelTrainTestPerformanceTestsConfig(BaseModel): Configuration for the train-test
                                                     performance comparison check.
    ModelTestsConfig(BaseModel): Root configuration for all model-related tests.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class ModelSimpleModelComparisonTestsConfig(BaseModel):
    """Configuration model for the SimpleModelComparison check.

    Attributes:
        min_allowed_gain (float): The minimum required performance gain over the
                                  simple model (>= 0.0).
        scorers (list[str]): List of metrics used for comparison.
        strategy (str): The strategy used by the simple baseline model.
    """

    min_allowed_gain: Annotated[float, Field(ge=0.0)]
    scorers: list[str]
    strategy: str


class ModelTrainTestPerformanceTestsConfig(BaseModel):
    """Configuration model for the TrainTestPerformance check.

    Attributes:
        degradation_threshold (float): The maximum allowed performance drop
                                       between train and test sets (in [0.0, 1.0]).
        test_min_score (float): The minimum required score on the test set (in [0.0, 1.0]).
        score (str): The main metric used for score comparison.
        scorers (list[str]): List of metrics used for performance evaluation.
    """

    degradation_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    test_min_score: Annotated[float, Field(ge=0.0, le=1.0)]
    score: str
    scorers: list[str]


class ModelTestsConfig(BaseModel):
    """Root configuration model for all model performance and robustness tests.

    Attributes:
        simple_model_comparison (ModelSimpleModelComparisonTestsConfig):
            Configuration for model comparison against a simple baseline.
        train_test_performance (ModelTrainTestPerformanceTestsConfig):
            Configuration for performance degradation checks.
    """

    simple_model_comparison: ModelSimpleModelComparisonTestsConfig
    train_test_performance: ModelTrainTestPerformanceTestsConfig
