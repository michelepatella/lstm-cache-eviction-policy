from pydantic import BaseModel, confloat


# Data — Split
class SplitConfig(BaseModel):
    training: confloat(ge=0, le=1)  # type: ignore[valid-type]
    validation: confloat(ge=0, le=1)  # type: ignore[valid-type]


class DatasetConfig(BaseModel):
    """
    Class representing the dataset configuration
    settings including split settings.
    """

    split: SplitConfig
