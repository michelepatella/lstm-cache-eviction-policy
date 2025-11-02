from pydantic import BaseModel, confloat


class DatasetSplitConfig(BaseModel):
    """Configuration for dataset split.

    Attributes:
        training (float): Fraction of the dataset used for training (in [0,1]).
        validation (float): Fraction of the dataset used for validation (in [0,1]).
    """

    training: confloat(ge=0, le=1)
    validation: confloat(ge=0, le=1)


class DatasetConfig(BaseModel):
    """Dataset configuration.

    Attributes:
        split (DatasetSplitConfig): Dataset split configuration.
    """

    split: DatasetSplitConfig
