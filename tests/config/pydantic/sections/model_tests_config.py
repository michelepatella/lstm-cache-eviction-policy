"""model_tests_config.py

Module defining the Pydantic models for configuring tests related to
model performance, robustness, and behavior.

These classes enforce validation rules for thresholds on metrics like AUC,
inference time, performance degradation, and bias checks.

Classes:
    ModelPerformanceSimpleModelComparisonTestsConfig(BaseModel):
        Configuration for the simple model comparison check.
    ModelPerformanceTrainTestPerformanceTestsConfig(BaseModel):
        Configuration for the train-test performance comparison check.
    ModelPerformanceConfusionMatrixReportTestsConfig(BaseModel):
        Configuration for the confusion matrix report check.
    ModelPerformanceTestsConfig(BaseModel): Configuration for the model performance.
    ModelTrainingReproducibilityTestsConfig(BaseModel): Configuration for model training
                                                        reproducibility check.
    ModelTrainingLearnabilityTestsConfig (BaseModel): Configuration for model training
                                                      learnability check.
    ModelTrainingTestsConfig(BaseModel): Configuration for the model training tests.
    ModelTestsConfig(BaseModel): Root configuration for all model-related tests.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class ModelPerformanceSimpleModelComparisonTestsConfig(BaseModel):
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


class ModelPerformanceTrainTestPerformanceTestsConfig(BaseModel):
    """Configuration model for the TrainTestPerformance check.

    Attributes:
        degradation_threshold (float): The maximum allowed performance drop
                                       between train and test sets (in [0.0, 1.0]).
        test_min_score (float): The minimum required score on the test set (in [0.0, 1.0]).
        scorers (list[str]): List of metrics used for performance evaluation.
    """

    degradation_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    test_min_score: Annotated[float, Field(ge=0.0, le=1.0)]
    scorers: list[str]


class ModelPerformanceConfusionMatrixReportTestsConfig(BaseModel):
    """Configuration model for the ConfusionMatrixReport check.

    Attributes:
        misclassified_samples_threshold (ModelPerformanceSimpleModelComparisonTestsConfig):
            Maximum allowed number misclassified in terms of percentage
            (in [0.0, 1.0]).
    """

    misclassified_samples_threshold: Annotated[float, Field(ge=0.0, le=1.0)]


class ModelPerformanceTestsConfig(BaseModel):
    """Configuration model for all model performanc tests.

    Attributes:
        simple_model_comparison (ModelPerformanceSimpleModelComparisonTestsConfig):
            Configuration for model comparison against a simple baseline.
        train_test_performance (ModelPerformanceTrainTestPerformanceTestsConfig):
            Configuration for performance degradation checks.
        confusion_matrix_report (ModelConfusionMatrixReportTestsConfig):
            Configuration for confusion matrix report check.
    """

    simple_model_comparison: ModelPerformanceSimpleModelComparisonTestsConfig
    train_test_performance: ModelPerformanceTrainTestPerformanceTestsConfig
    confusion_matrix_report: ModelPerformanceConfusionMatrixReportTestsConfig


class ModelTrainingReproducibilityTestsConfig(BaseModel):
    """Configuration model for model training reproducibility test.

    Attributes:
        avg_loss_tolerance (float): Tolerance difference among average losses
                                    obtained from the same seed-set training
                                    processes.
    """

    avg_loss_tolerance: Annotated[float, Field(ge=0.0)]


class ModelTrainingLearnabilityTestsConfig(BaseModel):
    """Configuration model for model training learnability test.

    Attributes:
        target_avg_loss (float): Expected average loss after training
                                 on a batch.
        avg_loss_tolerance (float): Tolerance w.r.t. expected average
                                    loss and observed one.
    """

    target_avg_loss: Annotated[float, Field(ge=0.0)]
    avg_loss_tolerance: Annotated[float, Field(ge=0.0)]


class ModelTrainingTestsConfig(BaseModel):
    """Configuration model for all model training tests.

    Attributes:
        learnability (ModelTrainingLearnabilityTestsConfig):
            Configuration for model training learnability tests.
        reproducibility (ModelTrainingReproducibilityTestsConfig):
            Configuration for model training reproducibility tests.
    """

    learnability: ModelTrainingLearnabilityTestsConfig
    reproducibility: ModelTrainingReproducibilityTestsConfig


class ModelTestsConfig(BaseModel):
    """Root configuration model for all model tests.

    Attributes:
        performance (ModelPerformanceTestsConfig):
            Configuration for model performance.
        training (ModelTrainingTestsConfig):
            Configuration for model training tests.
    """

    performance: ModelPerformanceTestsConfig
    training: ModelTrainingTestsConfig
