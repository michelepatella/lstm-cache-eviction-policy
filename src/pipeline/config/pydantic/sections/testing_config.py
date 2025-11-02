from pydantic import BaseModel, conint


class TestingGeneralConfig(BaseModel):
    """General configuration for testing.

    Attributes:
        batch_size (int): Number of samples per batch (> 0).
        shuffle (bool): Whether to shuffle the dataset during testing.
    """

    batch_size: conint(gt=0)
    shuffle: bool


class TestingMetricsConfig(BaseModel):
    """Metrics configuration for testing.

    Attributes:
        top_k (int): Number of top predictions to consider for evaluation (> 0).
    """

    top_k: conint(gt=0)


class TestingConfig(BaseModel):
    """Testing configuration.

    Attributes:
        general (TestingGeneralConfig): General testing configuration.
        metrics (TestingMetricsConfig): Testing metrics configuration.
    """

    general: TestingGeneralConfig
    metrics: TestingMetricsConfig
