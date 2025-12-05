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
from tests.config.pydantic.sections.seed_tests_config import SeedTestsConfig


class TestsConfig(BaseModel):
    """The root configuration model for the tests.

    Attributes:
        data (DataTestsConfig): Configuration for all data tests.
        seed (SeedTestsConfig): Configuration for the global seed used during
                                testing.
    """

    data: DataTestsConfig
    seed: SeedTestsConfig
