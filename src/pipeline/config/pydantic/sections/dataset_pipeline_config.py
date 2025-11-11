"""dataset_pipeline_config.py

Configuration section for dataset processing steps.

This module defines parameters for splitting the dataset (training/validation
fractions) and initial data cleaning procedures, such as handling missing
values.

Classes:
    DatasetSplitsPipelineConfig(BaseModel):
        Configuration for dataset split.
    DatasetCleaningMissingValuesRemovalDropnaPipelineConfig(BaseModel):
        Configuration for 'dropna' method.
    DatasetCleaningMissingValuesRemovalPipelineConfig(BaseModel):
        Configuration for missing values removal.
    DatasetCleaningPipelineConfig(BaseModel):
        Configuration for data cleaning steps.
    DatasetPipelineConfig(BaseModel):
        Aggregates all dataset configuration settings.
"""

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from components.assertions.choice_field_assertor import assert_choice_field
from pipeline.const import DATASET_MISSING_VALUES_REMOVAL_DROPNA_HOWS


class DatasetSplitsPipelineConfig(BaseModel):
    """Configuration for dataset split.

    Attributes:
        training (float): Fraction of the dataset used for training
                          (in [0,1]).
        validation (float): Fraction of the dataset used for validation
                            (in [0,1]).
    """

    training: Annotated[float, Field(ge=0, le=1)]
    validation: Annotated[float, Field(ge=0, le=1)]


class DatasetCleaningMissingValuesRemovalDropnaPipelineConfig(BaseModel):
    """Configuration for pandas dropna function.

    Attributes:
        how (str): Determine if row is removed.
    """

    how: str

    @model_validator(mode="after")
    def check_dataset_cleaning_missing_values_removal_dropna_how(
        self: "DatasetCleaningMissingValuesRemovalDropnaPipelineConfig",
    ) -> "DatasetCleaningMissingValuesRemovalDropnaPipelineConfig":
        """Check whether dropna how for dataset missing values removal
        is valid or not.

        This function validates the dropna how for dataset missing
        values removal.

        Args:
            self (DatasetCleaningMissingValuesRemovalDropnaPipelineConfig):
                Current model instance.

        Returns:
            "DatasetCleaningMissingValuesRemovalDropnaConfig":
                Validated model instance.
        """
        assert_choice_field(
            self.how,
            DATASET_MISSING_VALUES_REMOVAL_DROPNA_HOWS,
            "dataset.cleaning.missing_values_removal.dropna.how",
        )

        return self


class DatasetCleaningMissingValuesRemovalPipelineConfig(BaseModel):
    """Configuration for missing values removal step.

    Attributes:
        dropna (DatasetCleaningMissingValuesRemovalDropnaPipelineConfig):
            Configuration for row dropping.
    """

    dropna: DatasetCleaningMissingValuesRemovalDropnaPipelineConfig


class DatasetCleaningPipelineConfig(BaseModel):
    """Configuration for data cleaning steps.

    Attributes:
        missing_values_removal (DatasetCleaningMissingValuesRemovalPipelineConfig):
            Missing values removal configuration.
    """

    missing_values_removal: DatasetCleaningMissingValuesRemovalPipelineConfig


class DatasetPipelineConfig(BaseModel):
    """Dataset configuration.

    Attributes:
        cleaning (DatasetCleaningPipelineConfig): Data cleaning configuration.
        splits (DatasetSplitsPipelineConfig): Dataset split configuration.
    """

    cleaning: DatasetCleaningPipelineConfig
    splits: DatasetSplitsPipelineConfig
