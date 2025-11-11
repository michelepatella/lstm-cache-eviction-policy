"""seed_tests_config.py

Module defining the Pydantic model for configuring a global seed for
tests.

This class ensures strict validation for the seed value for ensuring test
reproducibility.

Classes:
    SeedTestsConfig: Configuration for the global random seed value.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class SeedTestsConfig(BaseModel):
    """Configuration model for the global random seed value.

    Attributes:
        value (int): The integer value used as the random seed (>= 0).
    """

    value: Annotated[int, Field(ge=0)]
