"""tests_config.py

Module defining the root Pydantic data model for the tests configuration.

This module combines sub-configurations defined in separate modules to create
a single, unified configuration structure for testing.

Classes:
    TestsConfig(BaseModel): The root configuration model, aggregating all
                            sub-configurations.
"""

from pydantic import BaseModel

from tests.config.pydantic.sections.data_tests_config import DataTestsConfig
from tests.config.pydantic.sections.model_tests_config import ModelTestsConfig
from tests.config.pydantic.sections.seed_tests_config import SeedTestsConfig
from tests.config.pydantic.sections.train_test_validation_tests_config import (
    TrainTestValidationTestsConfig,
)


class TestsConfig(BaseModel):
    """The root configuration model for the tests.

    Attributes:
        data (DataTestsConfig): Configuration for all data tests.
        model (ModelTestsConfig): Configuration for all model performance and
                                  robustness tests.
        seed (SeedTestsConfig): Configuration for the global seed used during
                                testing.
        train_test_validation (TrainTestValidationTestsConfig):
            Configuration for checks between data splits.
    """

    data: DataTestsConfig
    model: ModelTestsConfig
    seed: SeedTestsConfig
    train_test_validation: TrainTestValidationTestsConfig
