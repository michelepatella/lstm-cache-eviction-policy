from pydantic import BaseModel, confloat, conint


# Model — General
class FeaturesConfig(BaseModel):
    count: conint(gt=0)


class GeneralModelConfig(BaseModel):
    features: FeaturesConfig
    path: str


# Model — Params
class ModelParamsConfig(BaseModel):
    hidden_size: conint(gt=0)
    num_layers: conint(gt=0)
    bias: bool
    batch_first: bool
    dropout: confloat(ge=0, lt=1)
    bidirectional: bool
    proj_size: conint(ge=0)


class ModelConfig(BaseModel):
    """
    Class representing the model configuration
    settings including general and model param settings.
    """

    general: GeneralModelConfig
    params: ModelParamsConfig
