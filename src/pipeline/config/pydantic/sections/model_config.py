from pydantic import BaseModel, confloat, conint


class ModelOptimizationPruningConfig(BaseModel):
    """Model optimization pruning configuration.

    Attributes:
        amount (float): Pruning amount to apply (in [0.0, 1.0]).
    """
    amount: confloat(ge=0.0, le=1.0)


class ModelOptimizationConfig(BaseModel):
    """Model optimization configuration.

    Attributes:
        pruning (ModelOptimizationPruningConfig): Pruning configuration.
    """

    pruning: ModelOptimizationPruningConfig


class ModelParamsConfig(BaseModel):
    """Model parameters' configuration.

    Attributes:
        hidden_size (int): Hidden layer size (> 0).
        num_layers (int): Number of LSTM layers (> 0).
        bias (bool): Whether model layers use bias.
        batch_first (bool): If True, input/output tensors are of shape
                            (batch, seq, feature).
        dropout (float): Dropout probability (in [0, 1)).
        bidirectional (bool): Whether model layers are bidirectional.
        proj_size (int): Output projection size (>= 0).
    """

    hidden_size: conint(gt=0)
    num_layers: conint(gt=0)
    bias: bool
    batch_first: bool
    dropout: confloat(ge=0, lt=1)
    bidirectional: bool
    proj_size: conint(ge=0)


class ModelSequenceEmbeddingConfig(BaseModel):
    """Embedding configuration for sequence input.

    Attributes:
        dimension (int): Dimension of the embedding vector (> 0).
    """

    dimension: conint(gt=0)


class ModelSequenceConfig(BaseModel):
    """Sequence configuration for the model.

    Attributes:
        length (int): Length of the input sequences (> 0).
        embedding (ModelSequenceEmbeddingConfig): Embedding configuration.
    """

    length: conint(gt=0)
    embedding: ModelSequenceEmbeddingConfig


class ModelConfig(BaseModel):
    """Model configuration.

    Attributes:
        params (ModelParamsConfig): Model layer configuration.
        sequence (ModelSequenceConfig): Sequence and embedding configuration.
        optimization (ModelOptimizationConfig): Model optimization configuration.
    """

    params: ModelParamsConfig
    sequence: ModelSequenceConfig
    optimization: ModelOptimizationConfig
