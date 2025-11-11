"""seed_pipeline_config.py

Configuration section for the global random seed.

This module defines the single integer value used to ensure reproducibility.

Classes:
    SeedPipelineConfig(BaseModel):
        Configuration for the global random seed.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class SeedPipelineConfig(BaseModel):
    """Global random seed configuration.

    Attributes:
        value (int): The integer value used to set the global random
                     seed (>= 0).
    """

    value: Annotated[int, Field(ge=0)]
