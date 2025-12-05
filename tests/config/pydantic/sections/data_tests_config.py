"""data_tests_config.py

Module defining Pydantic models for configuring data tests
across raw and processed datasets.

These classes enforce strict validation rules for various data quality checks,
ensuring consistent configuration throughout  the testing phase of the pipeline.

Classes:
    DataIntegrityFeatureLabelCorrelationTestsConfig: Configuration for feature-label
                                                     correlation check.
    DataIntegrityFeatureFeatureCorrelationTestsConfig: Configuration for feature-feature
                                                       correlation check.
    DataIntegrityPercentOfNullsTestsConfig: Configuration for percentage of nulls check.
    DataIntegrityTestsConfig: Aggregates all specific data integrity test configurations.
    DataRawProcessedTestsConfig: Configuration for tests on a single data type
                                 (raw/processed).
    DataTestsConfig: Root configuration aggregating tests for data.
"""

from typing import Annotated

from pydantic import BaseModel, Field, confloat


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
    """

    processed: DataRawProcessedTestsConfig
    raw: DataRawProcessedTestsConfig
