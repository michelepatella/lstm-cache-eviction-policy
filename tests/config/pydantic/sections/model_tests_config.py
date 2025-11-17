"""model_tests_config.py

Module defining the Pydantic models for configuring tests related to
model performance.

These classes enforce validation rules for thresholds on metrics.

Classes:
    ModelPerformanceBaselineCompTestsConfig(BaseModel): Configuration for the simple
                                                        model comparison check.
    ModelPerformanceTrainTestTestsConfig(BaseModel): Configuration for the train-test
                                                     performance comparison check.
    ModelPerformanceConfMatrixTestsConfig(BaseModel): Configuration for the confusion
                                                      matrix report check.
    ModelPerformanceTestsConfig(BaseModel): Configuration for the model performance.
    ModelInferenceLatencyTestsConfig(BaseModel): Configuration for the model inference
                                                 latency check.
    ModelInferenceThroughputTestsConfig(BaseModel): Configuration for the model inference
                                                    throughput check.
    ModelInferenceTestsConfig(BaseModel): Configuration for the model inference tests.
    ModelTrainingLearnabilityLossTestsConfig(BaseModel): Configuration for model training
                                                         learnability loss target/tolerance.
    ModelTrainingLearnabilityTestsConfig (BaseModel): Configuration for model training
                                                      learnability check.
    ModelTrainingTestsConfig(BaseModel): Configuration for the model training tests.
    ModelTestsConfig(BaseModel): Root configuration for all model-related tests.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class ModelPerformanceBaselineCompTestsConfig(BaseModel):
    """Configuration model for the SimpleModelComparison check.

    Attributes:
        min_gain (float): The minimum required performance gain over the
                          simple model (>= 0.0).
        scorers (list[str]): List of metrics used for comparison.
        strategy (str): The strategy used by the simple baseline model.
    """

    min_gain: Annotated[float, Field(ge=0.0)]
    scorers: list[str]
    strategy: str


class ModelPerformanceTrainTestTestsConfig(BaseModel):
    """Configuration model for the TrainTestPerformance check.

    Attributes:
        max_degradation (float): The maximum allowed performance drop
                                 between train and test sets (in [0.0, 1.0]).
        test_min_score (float): The minimum required score on the test set
                                (in [0.0, 1.0]).
        scorers (list[str]): List of metrics used for performance evaluation.
    """

    max_degradation: Annotated[float, Field(ge=0.0, le=1.0)]
    test_min_score: Annotated[float, Field(ge=0.0, le=1.0)]
    scorers: list[str]


class ModelPerformanceConfMatrixTestsConfig(BaseModel):
    """Configuration model for the ConfusionMatrixReport check.

    Attributes:
        max_misclass (float): Maximum allowed number misclassified in terms
                              of percentage (in [0.0, 1.0]).
    """

    max_misclass: Annotated[float, Field(ge=0.0, le=1.0)]


class ModelPerformanceTestsConfig(BaseModel):
    """Configuration model for all model performance tests.

    Attributes:
        baseline_comp (ModelPerformanceBaselineCompTestsConfig):
            Configuration for model comparison against a simple baseline.
        train_test (ModelPerformanceTrainTestTestsConfig):
            Configuration for performance degradation checks.
        conf_matrix (ModelPerformanceConfMatrixTestsConfig):
            Configuration for confusion matrix report check.
    """

    baseline_comp: ModelPerformanceBaselineCompTestsConfig
    train_test: ModelPerformanceTrainTestTestsConfig
    conf_matrix: ModelPerformanceConfMatrixTestsConfig


class ModelInferenceLatencyTestsConfig(BaseModel):
    """Configuration model for model inference average latency check.

    Attributes:
        target_avg (float): The maximum allowed average latency in seconds
                            (>= 0.0).
        tol (float): The tolerance difference w.r.t. the target average
                     latency (>= 0.0).
    """

    target_avg: Annotated[float, Field(ge=0.0)]
    tol: Annotated[float, Field(ge=0.0)]


class ModelInferenceThroughputTestsConfig(BaseModel):
    """Configuration model for model inference throughput check.

    Attributes:
        min (int): The minimum required throughput (samples per second) (> 0).
    """

    min: Annotated[int, Field(gt=0)]


class ModelInferenceTestsConfig(BaseModel):
    """Configuration model for all model inference tests.

    Attributes:
        latency (ModelInferenceLatencyTestsConfig): Configuration for model inference
                                                    latency checks.
        throughput (ModelInferenceThroughputTestsConfig): Configuration for model
                                                          inference throughput checks.
    """

    latency: ModelInferenceLatencyTestsConfig
    throughput: ModelInferenceThroughputTestsConfig


class ModelTrainingLearnabilityLossTestsConfig(BaseModel):
    """Configuration model for model training learnability loss settings.

    Attributes:
        target_avg (float): Expected average loss after training
                            on a batch (>= 0.0).
        tol (float): Tolerance w.r.t. expected average loss and observed one
                     (>= 0.0).
    """

    target_avg: Annotated[float, Field(ge=0.0)]
    tol: Annotated[float, Field(ge=0.0)]


class ModelTrainingLearnabilityTestsConfig(BaseModel):
    """Configuration model for model training learnability test.

    Attributes:
        loss (ModelTrainingLearnabilityLossTestsConfig): Configuration for
                                                         expected loss values.
    """

    loss: ModelTrainingLearnabilityLossTestsConfig


class ModelTrainingTestsConfig(BaseModel):
    """Configuration model for all model training tests.

    Attributes:
        learnability (ModelTrainingLearnabilityTestsConfig):
            Configuration for model training learnability tests.
    """

    learnability: ModelTrainingLearnabilityTestsConfig


class ModelTestsConfig(BaseModel):
    """Root configuration model for all model tests.

    Attributes:
        inference (ModelInferenceTestsConfig): Configuration for model inference tests
                                               (latency, throughput).
        performance (ModelPerformanceTestsConfig): Configuration for model performance
                                                   checks (baseline, train/test, conf matrix).
        training (ModelTrainingTestsConfig): Configuration for model training tests
                                             (learnability).
    """

    inference: ModelInferenceTestsConfig
    performance: ModelPerformanceTestsConfig
    training: ModelTrainingTestsConfig
