from pydantic import BaseModel, confloat, conint


class GeneralModelConfig(BaseModel):
    """
    General configuration for the model.

    Attributes:
        features (int): Number of input features for the model (> 0).
    """

    features: conint(gt=0)  # type: ignore[valid-type]


class ModelParamsConfig(BaseModel):
    """
    Model parameters' configuration.

    Attributes:
        hidden_size (int): Hidden layer size (> 0).
        num_layers (int): Number of LSTM layers (> 0).
        bias (bool): Whether LSTM layers use bias.
        batch_first (bool): If True, input/output tensors are of shape
                            (batch, seq, feature).
        dropout (float): Dropout probability (in [0, 1)).
        bidirectional (bool): Whether LSTM layers are bidirectional.
        proj_size (int): Output projection size (>= 0).
    """

    hidden_size: conint(gt=0)  # type: ignore[valid-type]
    num_layers: conint(gt=0)  # type: ignore[valid-type]
    bias: bool
    batch_first: bool
    dropout: confloat(ge=0, lt=1)  # type: ignore[valid-type]
    bidirectional: bool
    proj_size: conint(ge=0)  # type: ignore[valid-type]


class EmbeddingConfig(BaseModel):
    """
    Embedding configuration for sequence input.

    Attributes:
        dimension (int): Dimension of the embedding vector (> 0).
    """

    dimension: conint(gt=0)  # type: ignore[valid-type]


class SequenceConfig(BaseModel):
    """
    Sequence configuration for the model.

    Attributes:
        length (int): Length of the input sequences (> 0).
        embedding (EmbeddingConfig): Embedding configuration.
    """

    length: conint(gt=0)  # type: ignore[valid-type]
    embedding: EmbeddingConfig


class ModelConfig(BaseModel):
    """
    Model configuration.

    Attributes:
        general (GeneralModelConfig): General model configuration.
        params (ModelParamsConfig): LSTM layer configuration.
        sequence (SequenceConfig): Sequence and embedding configuration.
    """

    general: GeneralModelConfig
    params: ModelParamsConfig
    sequence: SequenceConfig
