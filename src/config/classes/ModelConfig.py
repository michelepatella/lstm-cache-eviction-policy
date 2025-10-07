from pydantic import BaseModel, confloat, conint


# Model — General
class GeneralModelConfig(BaseModel):
    features: conint(gt=0)  # type: ignore[valid-type]


# Model — Params
class ModelParamsConfig(BaseModel):
    hidden_size: conint(gt=0)  # type: ignore[valid-type]
    num_layers: conint(gt=0)  # type: ignore[valid-type]
    bias: bool
    batch_first: bool
    dropout: confloat(ge=0, lt=1)  # type: ignore[valid-type]
    bidirectional: bool
    proj_size: conint(ge=0)  # type: ignore[valid-type]


# Model — Sequence
class EmbeddingConfig(BaseModel):
    dimension: conint(gt=0)  # type: ignore[valid-type]


class SequenceConfig(BaseModel):
    length: conint(gt=0)  # type: ignore[valid-type]
    embedding: EmbeddingConfig


class ModelConfig(BaseModel):
    """
    Class representing the model configuration
    settings including general and model param settings.
    """

    general: GeneralModelConfig
    params: ModelParamsConfig
    sequence: SequenceConfig
