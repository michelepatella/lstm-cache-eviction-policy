"""data_tests_config.py

Module defining Pydantic models for configuring data tests
across raw, processed, and training-testing-validation datasets.

These classes enforce strict validation rules for various data quality checks,
ensuring consistent configuration throughout the testing phase of the pipeline.

Classes:
    DataTrainTestValidationIndexLeakageTestsConfig(BaseModel):
        Configuration for the index leakage check.
    DataTrainTestValidationFeatureLabelCorrelationChangeTestsConfig(BaseModel):
        Configuration for the feature-label correlation change check.
    DataTrainTestValidationMultivariateDriftTestsConfig(BaseModel):
        Configuration for the multivariate drift check.
    DataTrainTestValidationLabelDriftTestsConfig(BaseModel):
        Configuration for the label drift check.
    DataTrainTestValidationFeatureDriftTestsConfig(BaseModel):
        Configuration for the feature drift check.
    DataTrainTestValidationDatasetsSizeComparisonTestsConfig(BaseModel):
        Configuration for the datasets size comparison check.
    DataTrainTestValidationTestsConfig(BaseModel): Root configuration for all
                                                   train/test/validation tests.
    DataIntegrityFeatureLabelCorrelationTestsConfig(BaseModel):
        Configuration for feature-label correlation check.
    DataIntegrityFeatureFeatureCorrelationTestsConfig(BaseModel):
        Configuration for feature-feature correlation check.
    DataIntegrityPercentOfNullsTestsConfig(BaseModel): Configuration for percentage of
                                                       nulls check.
    DataIntegrityTestsConfig(BaseModel): Aggregates all specific data integrity test
                                         configurations.
    DataRawProcessedTestsConfig(BaseModel): Configuration for tests on a single data
                                            type (raw/processed).
    DataTestsConfig(BaseModel): Root configuration aggregating tests for data.
"""

from typing import Annotated

from pydantic import BaseModel, Field, confloat


class DataTrainTestValidationIndexLeakageTestsConfig(BaseModel):
    """Configuration model for the IndexTrainTestLeakage check.

    Attributes:
        max_ratio (float): The maximum allowed ratio of shared indices
                           between data splits (in [0.0, 1.0]).
    """

    max_ratio: Annotated[float, Field(ge=0.0, le=1.0)]


class DataTrainTestValidationFeatureLabelCorrelationChangeTestsConfig(
    BaseModel,
):
    """Configuration model for the FeatureLabelCorrelationChange check.

    Attributes:
        threshold (float): The maximum allowed difference in feature-label
                           correlation (in [0.0, 1.0]).
    """

    threshold: Annotated[float, Field(ge=0.0, le=1.0)]


class DataTrainTestValidationMultivariateDriftTestsConfig(BaseModel):
    """Configuration model for the MultivariateDrift check.

    Attributes:
        max_drift_value (float): The maximum allowed overall drift value
                                 (>= 0.0).
    """

    max_drift_value: Annotated[float, Field(ge=0.0)]


class DataTrainTestValidationLabelDriftTestsConfig(BaseModel):
    """Configuration model for the LabelDrift check.

    Attributes:
        max_allowed_drift_score (float): The maximum allowed drift score for the
                                         target variable (>= 0.0).
    """

    max_allowed_drift_score: Annotated[float, Field(ge=0.0)]


class DataTrainTestValidationFeatureDriftTestsConfig(BaseModel):
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


class DataTrainTestValidationDatasetsSizeComparisonTestsConfig(BaseModel):
    """Configuration model for the DatasetsSizeComparison check.

    Attributes:
        ratio (float): The minimum expected ratio between the test/validation
                       set size and the training set size (in [0.0, 1.0]).
    """

    ratio: Annotated[float, Field(ge=0.0, le=1.0)]


class DataTrainTestValidationTestsConfig(BaseModel):
    """Root configuration model for all data splits consistency
    and drift tests.

    Attributes:
        index_leakage (DataTrainTestValidationIndexLeakageTestsConfig):
            Configuration for training-testing index leakage check.
        feature_label_correlation_change (DataTrainTestValidationFeatureLabelCorrelationChangeTestsConfig):
                Configuration for correlation stability.
        multivariate_drift (DataTrainTestValidationMultivariateDriftTestsConfig):
            Configuration for overall multivariate drift.
        label_drift (DataTrainTestValidationLabelDriftTestsConfig):
            Configuration for target variable drift.
        feature_drift (DataTrainTestValidationFeatureDriftTestsConfig):
            Configuration for individual feature drift.
        datasets_size_comparison (DataTrainTestValidationDatasetsSizeComparisonTestsConfig):
            Configuration for checking the size ratio between training and other splits.
    """

    index_leakage: DataTrainTestValidationIndexLeakageTestsConfig
    feature_label_correlation_change: (
        DataTrainTestValidationFeatureLabelCorrelationChangeTestsConfig
    )
    multivariate_drift: DataTrainTestValidationMultivariateDriftTestsConfig
    label_drift: DataTrainTestValidationLabelDriftTestsConfig
    feature_drift: DataTrainTestValidationFeatureDriftTestsConfig
    datasets_size_comparison: (
        DataTrainTestValidationDatasetsSizeComparisonTestsConfig
    )


class DataIntegrityFeatureLabelCorrelationTestsConfig(BaseModel):
    """Configuration model for feature-label correlation test.

    Attributes:
        threshold (float): The maximum allowed absolute correlation value
                           before flagging a failed test (in [0.0, 1.0]).
    """

    threshold: confloat(ge=0.0, le=1.0)


class DataIntegrityFeatureFeatureCorrelationTestsConfig(BaseModel):
    """Configuration model for feature-feature correlation test.

    Attributes:
        num_pairs (int): The number of top correlated feature pairs to check.
        threshold (float): The maximum allowed absolute correlation value
                           (in [0.0, 1.0]).
    """

    num_pairs: Annotated[int, Field(ge=0)]
    threshold: Annotated[float, Field(ge=0.0, le=1.0)]


class DataIntegrityPercentOfNullsTestsConfig(BaseModel):
    """Configuration model for the percentage of null values test.

    Attributes:
        threshold (float): The maximum allowed percentage of null values
                           in any feature column (in [0.0, 1.0]).
    """

    threshold: Annotated[float, Field(ge=0.0, le=1.0)]


class DataIntegrityTestsConfig(BaseModel):
    """Aggregated configuration model for all data integrity tests.

    Attributes:
        feature_feature_correlation (DataIntegrityFeatureFeatureCorrelationTestsConfig):
            Configuration for feature-feature correlation.
        feature_label_correlation (DataIntegrityFeatureLabelCorrelationTestsConfig):
            Configuration for feature-label correlation.
        percent_of_nulls (DataIntegrityPercentOfNullsTestsConfig):
            Configuration for null check.
    """

    feature_feature_correlation: (
        DataIntegrityFeatureFeatureCorrelationTestsConfig
    )
    feature_label_correlation: DataIntegrityFeatureLabelCorrelationTestsConfig
    percent_of_nulls: DataIntegrityPercentOfNullsTestsConfig


class DataRawProcessedTestsConfig(BaseModel):
    """Configuration model for tests applied to a data type (raw or processed).

    Attributes:
        integrity (DataIntegrityTestsConfig): The set of integrity tests to run.
    """

    integrity: DataIntegrityTestsConfig


class DataTestsConfig(BaseModel):
    """The root configuration model for all data tests.

    Attributes:
        processed (DataRawProcessedTestsConfig): Configuration for tests on the
                                                 processed dataset.
        raw (DataRawProcessedTestsConfig): Configuration for tests on the raw dataset.
        train_test_validation (train_test_validation): Configuration for training, testing
                                                       and validation sets.
    """

    processed: DataRawProcessedTestsConfig
    raw: DataRawProcessedTestsConfig
    train_test_validation: DataTrainTestValidationTestsConfig
