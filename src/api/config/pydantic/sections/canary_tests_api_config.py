"""canary_tests_api_config.py

Module defining Pydantic schemas for configuring canary tests.

These schemas ensure that canary test parameters are properly validated
and typed.

Classes:
    CanaryTestsAPIConfig(BaseModel):
        Configuration schema for controlling canary test behavior.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class CanaryTestsAPIConfig(BaseModel):
    """Canary tests configuration.

    Attributes:
        traffic_threshold (float): Fraction of traffic to route to the
                                   staging model.
    """

    traffic_threshold: Annotated[float, Field(ge=0.0, le=1.0)]
