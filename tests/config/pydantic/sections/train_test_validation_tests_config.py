"""train_test_validation_tests_config.py

Module defining the Pydantic models for configuring tests related to
consistency and drift between the training, validation, and testing datasets.

These classes enforce validation rules for thresholds on data quality checks
across data splits, including drift metrics and leakage prevention.

Classes:
    TrainTestValidationNewLabelTestsConfig(BaseModel): Configuration for the new label
                                                       check.
    TrainTestValidationIndexLeakageTestsConfig(BaseModel): Configuration for the index
                                                           leakage check.
    TrainTestValidationFeatureLabelCorrelationChangeTestsConfig(BaseModel):
        Configuration for the feature-label correlation change check.
    TrainTestValidationMultivariateDriftTestsConfig(BaseModel): Configuration for the
                                                                multivariate drift check.
    TrainTestValidationLabelDriftTestsConfig(BaseModel): Configuration for the label drift
                                                         check.
    TrainTestValidationFeatureDriftTestsConfig(BaseModel): Configuration for the feature
                                                          drift check.
    TrainTestValidationDatasetsSizeComparisonTestsConfig(BaseModel): Configuration for the
                                                                     datasets size comparison
                                                                     check.
    TrainTestValidationTestsConfig(BaseModel): Root configuration for all train/test/validation
                                               tests.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class TrainTestValidationNewLabelTestsConfig(BaseModel):
    """Configuration model for the NewLabelTrainTest check.

    Attributes:
        max_ratio (float): The maximum allowed ratio of new labels in the
                           test/validation sets (in [0.0, 1.0]).
    """

    max_ratio: Annotated[float, Field(ge=0.0, le=1.0)]


class TrainTestValidationIndexLeakageTestsConfig(BaseModel):
    """Configuration model for the IndexTrainTestLeakage check.

    Attributes:
        max_ratio (float): The maximum allowed ratio of shared indices
                           between data splits (in [0.0, 1.0]).
    """

    max_ratio: Annotated[float, Field(ge=0.0, le=1.0)]


class TrainTestValidationFeatureLabelCorrelationChangeTestsConfig(BaseModel):
    """Configuration model for the FeatureLabelCorrelationChange check.

    Attributes:
        threshold (float): The maximum allowed difference in feature-label
                           correlation (in [0.0, 1.0]).
    """

    threshold: Annotated[float, Field(ge=0.0, le=1.0)]


class TrainTestValidationMultivariateDriftTestsConfig(BaseModel):
    """Configuration model for the MultivariateDrift check.

    Attributes:
        max_drift_value (float): The maximum allowed overall drift value
                                 (>= 0.0).
    """

    max_drift_value: Annotated[float, Field(ge=0.0)]


class TrainTestValidationLabelDriftTestsConfig(BaseModel):
    """Configuration model for the LabelDrift check.

    Attributes:
        max_allowed_drift_score (float): The maximum allowed drift score for the
                                         target variable (>= 0.0).
    """

    max_allowed_drift_score: Annotated[float, Field(ge=0.0)]


class TrainTestValidationFeatureDriftTestsConfig(BaseModel):
    """Configuration model for the FeatureDrift check.

    Attributes:
        max_allowed_numeric_score (float): The maximum allowed drift score for
                                           individual numeric features (>= 0.0).
        allowed_num_features_exceeding_threshold (int): The maximum number of
                                                        features allowed to exceed
                                                        the drift score threshold
                                                        (>= 0).
    """

    max_allowed_numeric_score: Annotated[float, Field(ge=0.0)]
    allowed_num_features_exceeding_threshold: Annotated[int, Field(ge=0)]


class TrainTestValidationDatasetsSizeComparisonTestsConfig(BaseModel):
    """Configuration model for the DatasetsSizeComparison check.

    Attributes:
        ratio (float): The minimum expected ratio between the test/validation
                       set size and the training set size (in [0.0, 1.0]).
    """

    ratio: Annotated[float, Field(ge=0.0, le=1.0)]


class TrainTestValidationTestsConfig(BaseModel):
    """Root configuration model for all data splits consistency
    and drift tests.

    Attributes:
        new_label (TrainTestValidationNewLabelTestsConfig):
            Configuration for new label appearance in testing/validation sets.
        index_leakage (TrainTestValidationIndexLeakageTestsConfig):
            Configuration for training-testing index leakage check.
        feature_label_correlation_change (TrainTestValidationFeatureLabelCorrelationChangeTestsConfig):
                Configuration for correlation stability.
        multivariate_drift (TrainTestValidationMultivariateDriftTestsConfig):
            Configuration for overall multivariate drift.
        label_drift (TrainTestValidationLabelDriftTestsConfig):
            Configuration for target variable drift.
        feature_drift (TrainTestValidationFeatureDriftTestsConfig):
            Configuration for individual feature drift.
        datasets_size_comparison (TrainTestValidationDatasetsSizeComparisonTestsConfig):
            Configuration for checking the size ratio between training and other splits.
    """

    new_label: TrainTestValidationNewLabelTestsConfig
    index_leakage: TrainTestValidationIndexLeakageTestsConfig
    feature_label_correlation_change: (
        TrainTestValidationFeatureLabelCorrelationChangeTestsConfig
    )
    multivariate_drift: TrainTestValidationMultivariateDriftTestsConfig
    label_drift: TrainTestValidationLabelDriftTestsConfig
    feature_drift: TrainTestValidationFeatureDriftTestsConfig
    datasets_size_comparison: (
        TrainTestValidationDatasetsSizeComparisonTestsConfig
    )
