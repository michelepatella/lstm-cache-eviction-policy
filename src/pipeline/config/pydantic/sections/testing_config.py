from pydantic import BaseModel, conint


class GeneralTestingConfig(BaseModel):
    """
    General configuration for testing.

    Attributes:
        batch_size (int): Number of samples per batch (> 0).
        shuffle (bool): Whether to shuffle the dataset during testing.
    """

    batch_size: conint(gt=0)  # type: ignore[valid-type]
    shuffle: bool


class MetricsConfig(BaseModel):
    """
    Metrics configuration for testing.

    Attributes:
        top_k (int): Number of top predictions to consider for evaluation (> 0).
    """

    top_k: conint(gt=0)  # type: ignore[valid-type]


class TestingConfig(BaseModel):
    """
    Testing configuration.

    Attributes:
        general (GeneralTestingConfig): General testing configuration.
        metrics (MetricsConfig): Testing metrics configuration.
    """

    general: GeneralTestingConfig
    metrics: MetricsConfig
