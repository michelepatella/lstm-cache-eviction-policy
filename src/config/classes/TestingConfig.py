from pydantic import BaseModel, conint


# Testing — General
class GeneralTestingConfig(BaseModel):
    batch_size: conint(gt=0)  # type: ignore[valid-type]
    shuffle: bool


# Testing — Metrics
class MetricsConfig(BaseModel):
    top_k: conint(gt=0)  # type: ignore[valid-type]


class TestingConfig(BaseModel):
    """
    Class representing the testing configuration
    settings including general and metrics settings.
    """

    general: GeneralTestingConfig
    metrics: MetricsConfig
