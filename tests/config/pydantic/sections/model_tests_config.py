"""model_tests_config.py

Module defining the Pydantic models for configuring tests related to
model performance, robustness, and behavior.

These classes enforce validation rules for thresholds on metrics like AUC,
inference time, performance degradation, and bias checks.

Classes:
    ModelConfusionMatrixReportTestsConfig(BaseModel): Configuration for the confusion
                                                      matrix report check.
    ModelInferenceTimeTestsConfig(BaseModel): Configuration for the inference time check.
    ModelPerformanceBiasTestsConfig(BaseModel): Configuration for the performance bias check.
    ModelRocReportTestsConfig(BaseModel): Configuration for the ROC report check.
    ModelSimpleModelComparisonTestsConfig(BaseModel): Configuration for the simple model
                                                      comparison check.
    ModelSingleDatasetPerformanceTestsConfig(BaseModel): Configuration for the single
                                                         dataset performance check.
    ModelTrainTestPerformanceTestsConfig(BaseModel): Configuration for the train-test
                                                     performance comparison check.
    ModelUnusedFeaturesTestsConfig(BaseModel): Configuration for the unused features check.
    ModelWeakSegmentsPerformanceTestsConfig(BaseModel): Configuration for the weak segments
                                                        performance check.
    ModelTestsConfig(BaseModel): Root configuration for all model-related tests.
"""

from typing import Annotated, Dict

from pydantic import BaseModel, Field, confloat


class ModelConfusionMatrixReportTestsConfig(BaseModel):
    """Configuration model for the ConfusionMatrixReport check.

    Attributes:
        misclassified_samples_threshold (float): The maximum allowed ratio of
                                                 misclassified samples (in [0.0, 1.0]).
    """

    misclassified_samples_threshold: Annotated[float, Field(ge=0.0, le=1.0)]


class ModelInferenceTimeTestsConfig(BaseModel):
    """Configuration model for the InferenceTime check.

    Attributes:
        threshold (float): The maximum allowed inference time in seconds (>= 0.0).
    """

    threshold: Annotated[float, Field(ge=0.0)]


class ModelPerformanceBiasTestsConfig(BaseModel):
    """Configuration model for the PerformanceBias check.

    Attributes:
        lower_bound (float): The minimum required score difference (>= 0.0)
                             between segments for a bias check.
        scorer (str): The metric used for performance comparison.
    """

    lower_bound: Annotated[float, Field(ge=0.0)]
    scorer: str


class ModelRocReportTestsConfig(BaseModel):
    """Configuration model for the RocReport check.

    Attributes:
        min_auc (float): The minimum required Area Under the Curve (AUC) score
                         (in [0.0, 1.0]).
    """

    min_auc: Annotated[float, Field(ge=0.0, le=1.0)]


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


class ModelSingleDatasetPerformanceTestsConfig(BaseModel):
    """Configuration model for the SingleDatasetPerformance check.

    Attributes:
        class_mode (str): Mode for aggregating results across classes.
        scorers (list[str]): List of metrics used for performance evaluation.
        threshold (float): The minimum required score for the model performance
                           (in [0.0, 1.0]).
    """

    class_mode: str
    scorers: list[str]
    threshold: confloat(ge=0.0, le=1.0)


class ModelTrainTestPerformanceTestsConfig(BaseModel):
    """Configuration model for the TrainTestPerformance check.

    Attributes:
        degradation_threshold (float): The maximum allowed performance drop
                                       between train and test sets (in [0.0, 1.0]).
        imbalance_threshold (float): The maximum allowed imbalance in class distribution
                                     (in [0.0, 1.0]).
        test_min_score (float): The minimum required score on the test set (in [0.0, 1.0]).
        score (str): The main metric used for score comparison.
        scorers (list[str]): List of metrics used for performance evaluation.
    """

    degradation_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    imbalance_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
    test_min_score: Annotated[float, Field(ge=0.0, le=1.0)]
    score: str
    scorers: list[str]


class ModelUnusedFeaturesTestsConfig(BaseModel):
    """Configuration model for the UnusedFeatures check.

    Attributes:
        max_high_variance_unused_features (int): The maximum number of unused features
                                                 with high variance allowed (>= 0).
    """

    max_high_variance_unused_features: Annotated[int, Field(ge=0)]


class ModelWeakSegmentsPerformanceTestsConfig(BaseModel):
    """Configuration model for the WeakSegmentsPerformance check.

    Attributes:
        alternative_scorer (Dict[str, str]): Dictionary mapping standard scorer to an
                                             alternative.
        max_ratio_change (float): The maximum allowed ratio change in performance across
                                  segments (in [0.0, 1.0]).
    """

    alternative_scorer: Dict[str, str]
    max_ratio_change: Annotated[float, Field(ge=0.0, le=1.0)]


class ModelTestsConfig(BaseModel):
    """Root configuration model for all model performance and robustness tests.

    Attributes:
        confusion_matrix_report (ModelConfusionMatrixReportTestsConfig):
            Configuration for misclassified sample checks.
        inference_time (ModelInferenceTimeTestsConfig): Configuration for inference
                                                        latency checks.
        performance_bias (ModelPerformanceBiasTestsConfig): Configuration for performance
                                                            bias checks.
        roc_report (ModelRocReportTestsConfig): Config for AUC and ROC curve analysis.
        simple_model_comparison (ModelSimpleModelComparisonTestsConfig):
            Configuration for model comparison against a simple baseline.
        single_dataset_performance (ModelSingleDatasetPerformanceTestsConfig):
            Configuration for minimum required performance on a single dataset.
        train_test_performance (ModelTrainTestPerformanceTestsConfig):
            Configuration for performance degradation checks.
        unused_features (ModelUnusedFeaturesTestsConfig): Configuration for detecting unused
                                                          high-variance features.
        weak_segments_performance (ModelWeakSegmentsPerformanceTestsConfig):
            Configuration for performance stability across weak segments.
    """

    confusion_matrix_report: ModelConfusionMatrixReportTestsConfig
    inference_time: ModelInferenceTimeTestsConfig
    performance_bias: ModelPerformanceBiasTestsConfig
    roc_report: ModelRocReportTestsConfig
    simple_model_comparison: ModelSimpleModelComparisonTestsConfig
    single_dataset_performance: ModelSingleDatasetPerformanceTestsConfig
    train_test_performance: ModelTrainTestPerformanceTestsConfig
    unused_features: ModelUnusedFeaturesTestsConfig
    weak_segments_performance: ModelWeakSegmentsPerformanceTestsConfig
