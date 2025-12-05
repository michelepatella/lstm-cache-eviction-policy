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


class TestsConfig(BaseModel):
    """The root configuration model for the tests.

    Attributes:
        data (DataTestsConfig): Configuration for all data tests.
        model (ModelTestsConfig): Configuration for all model tests
    """

    data: DataTestsConfig
    model: ModelTestsConfig
