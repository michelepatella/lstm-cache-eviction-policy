"""data_tests_config.py

Module defining Pydantic models for configuring data tests
across raw, processed, and training-testing-validation datasets.

These classes enforce strict validation rules for various data quality checks,
ensuring consistent configuration throughout the testing phase of the pipeline.

Classes:
    DataTrainTestValidationIndexLeakageTestsConfig(BaseModel):
        Configuration for the index leakage check.
    DataTrainTestValidationFeatLabelCorrChangeTestsConfig(BaseModel):
        Configuration for the feature-label correlation change check.
    DataTrainTestValidationMultivariateDriftTestsConfig(BaseModel):
        Configuration for the multivariate drift check.
    DataTrainTestValidationLabelDriftTestsConfig(BaseModel):
        Configuration for the label drift check.
    DataTrainTestValidationFeatDriftTestsConfig(BaseModel):
        Configuration for the feature drift check.
    DataTrainTestValidationRatioTestsConfig(BaseModel):
        Configuration for the datasets size comparison check.
    DataTrainTestValidationTestsConfig(BaseModel): Root configuration for all
                                                   train/test/validation tests.
        DataIntegrityFeatLabelCorrTestsConfig(BaseModel):
        Configuration for feature-label correlation check.
    DataIntegrityFeatFeatCorrTestsConfig(BaseModel): Configuration for feature-feature
                                                     correlation check.
    DataIntegrityNullsPercTestsConfig(BaseModel): Configuration for percentage of
                                                  nulls check.

    DataIntegrityRawProcessedTestsConfig(BaseModel): Configuration for tests on integrity
                                                    of a single data type (raw/processed).
    DataIntegrityTestsConfig(BaseModel): Aggregates integrity tests for raw and processed
                                         data.
    DataTestsConfig(BaseModel): Root configuration aggregating all data tests.
"""

from typing import Annotated

from pydantic import BaseModel, Field, confloat


class DataTrainTestValidationIndexLeakageTestsConfig(BaseModel):
    """Configuration model for the IndexTrainTestLeakage check.

    Attributes:
        max (float): The maximum allowed ratio of shared indices between data
                     splits (in [0.0, 1.0]).
    """

    max: Annotated[float, Field(ge=0.0, le=1.0)]


class DataTrainTestValidationFeatLabelCorrChangeTestsConfig(
    BaseModel,
):
    """Configuration model for the FeatureLabelCorrelationChange check.

    Attributes:
        max (float): The maximum allowed difference in feature-label
                     correlation (in [0.0, 1.0]).
    """

    max: Annotated[float, Field(ge=0.0, le=1.0)]


class DataTrainTestValidationMultivariateDriftTestsConfig(BaseModel):
    """Configuration model for the MultivariateDrift check.

    Attributes:
        max (float): The maximum allowed overall drift value (>= 0.0).
    """

    max: Annotated[float, Field(ge=0.0)]


class DataTrainTestValidationLabelDriftTestsConfig(BaseModel):
    """Configuration model for the LabelDrift check.

    Attributes:
        max (float): The maximum allowed drift score for the target variable
                     (>= 0.0).
    """

    max: Annotated[float, Field(ge=0.0)]


class DataTrainTestValidationFeatDriftTestsConfig(BaseModel):
    """Configuration model for the FeatDrift check.

    Attributes:
        max (float): The maximum allowed drift score for
                     individual numeric features (>= 0.0).
        max_exceeding_feat (int): The maximum number of features allowed to exceed
                                  the drift score threshold (>= 0).
    """

    max: Annotated[float, Field(ge=0.0)]
    max_exceeding_feat: Annotated[int, Field(ge=0)]


class DataTrainTestValidationRatioTestsConfig(BaseModel):
    """Configuration model for the Ratio check.

    Attributes:
        min (float): The minimum expected ratio between the test/validation
                     set size and the training set size (in [0.0, 1.0]).
    """

    min: Annotated[float, Field(ge=0.0, le=1.0)]


class DataTrainTestValidationTestsConfig(BaseModel):
    """Root configuration model for all data splits consistency
    and drift tests.

    Attributes:
        index_leakage (DataTrainTestValidationIndexLeakageTestsConfig):
            Configuration for training-testing index leakage check.
        feat_label_corr_change (DataTrainTestValidationFeatLabelCorrChangeTestsConfig):
                Configuration for correlation stability.
        multivariate_drift (DataTrainTestValidationMultivariateDriftTestsConfig):
            Configuration for overall multivariate drift.
        label_drift (DataTrainTestValidationLabelDriftTestsConfig):
            Configuration for target variable drift.
        feat_drift (DataTrainTestValidationFeatDriftTestsConfig):
            Configuration for individual feature drift.
        ratio (DataTrainTestValidationRatioTestsConfig):
            Configuration for checking the size ratio between training and other splits.
    """

    index_leakage: DataTrainTestValidationIndexLeakageTestsConfig
    feat_label_corr_change: (
        DataTrainTestValidationFeatLabelCorrChangeTestsConfig
    )
    multivariate_drift: DataTrainTestValidationMultivariateDriftTestsConfig
    label_drift: DataTrainTestValidationLabelDriftTestsConfig
    feat_drift: DataTrainTestValidationFeatDriftTestsConfig
    ratio: DataTrainTestValidationRatioTestsConfig


class DataIntegrityFeatLabelCorrTestsConfig(BaseModel):
    """Configuration model for feature-label correlation test.

    Attributes:
        max (float): The maximum allowed absolute correlation value
                     before flagging a failed test (in [0.0, 1.0]).
    """

    max: confloat(ge=0.0, le=1.0)


class DataIntegrityFeatFeatCorrTestsConfig(BaseModel):
    """Configuration model for feature-feature correlation test.

    Attributes:
        max_exceeding_pairs (int): The maximum number of feature pairs that can
                                   exceed the maximum predefined value.
        max (float): The maximum allowed absolute correlation value
                     (in [0.0, 1.0]).
    """

    max_exceeding_pairs: Annotated[int, Field(ge=0)]
    max: Annotated[float, Field(ge=0.0, le=1.0)]


class DataIntegrityNullsPercTestsConfig(BaseModel):
    """Configuration model for the percentage of null values test.

    Attributes:
        max (float): The maximum allowed percentage of null values
                     in any feature column (in [0.0, 1.0]).
    """

    max: Annotated[float, Field(ge=0.0, le=1.0)]


class DataIntegrityRawProcessedTestsConfig(BaseModel):
    """Configuration model for integrity tests applied to a data type
    (raw or processed).

    This model groups all basic integrity checks for a single dataset type.

    Attributes:
        feat_feat_corr (DataIntegrityFeatFeatCorrTestsConfig):
            Configuration for feature-feature correlation.
        feat_label_corr (DataIntegrityFeatLabelCorrTestsConfig):
            Configuration for feature-label correlation.
        nulls_perc (DataIntegrityNullsPercTestsConfig):
            Configuration for null check.
    """

    feat_feat_corr: DataIntegrityFeatFeatCorrTestsConfig
    feat_label_corr: DataIntegrityFeatLabelCorrTestsConfig
    nulls_perc: DataIntegrityNullsPercTestsConfig


class DataIntegrityTestsConfig(BaseModel):
    """Root configuration model for all data integrity tests.

    This model contains the specific integrity checks for both the raw and
    the processed datasets.

    Attributes:
        processed (DataIntegrityRawProcessedTestsConfig): Configuration for integrity tests
                                                          on the processed dataset.
        raw (DataIntegrityRawProcessedTestsConfig): Configuration for integrity tests on
                                                    the raw dataset.
    """

    processed: DataIntegrityRawProcessedTestsConfig
    raw: DataIntegrityRawProcessedTestsConfig


class DataTestsConfig(BaseModel):
    """The root configuration model for all data tests.

    Attributes:
        integrity (DataIntegrityTestsConfig): Configuration for integrity checks
                                              on raw and processed data.
        train_test_validation (DataTrainTestValidationTestsConfig):
            Configuration for consistency checks between training, testing,
            and validation sets.
    """

    integrity: DataIntegrityTestsConfig
    train_test_validation: DataTrainTestValidationTestsConfig
