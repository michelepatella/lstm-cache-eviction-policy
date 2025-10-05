from pydantic import BaseModel, conint


class TestingConfig(BaseModel):
    """
    Class representing the testing configuration
    settings.
    """

    batch_size: conint(gt=0)  # type: ignore[valid-type]
    shuffle: bool
