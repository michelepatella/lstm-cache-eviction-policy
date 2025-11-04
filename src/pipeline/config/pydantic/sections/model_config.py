"""model_config.py

Configuration section for the model's architecture,
sequence handling, and optimizations.

This module defines the core parameters of the model, sequence processing
(embedding, length), and post-training optimizations (pruning and quantization).

Classes:
    ModelOptimizationsPruningConfig(BaseModel):
        Pruning configuration.
    ModelOptimizationsQuantizationConfig(BaseModel):
        Quantization configuration.
    ModelOptimizationsConfig(BaseModel):
        Aggregates model optimization settings.
    ModelParamsConfig(BaseModel):
        Model parameters' configuration.
    ModelSequenceEmbeddingConfig(BaseModel):
        Embedding configuration.
    ModelSequenceConfig(BaseModel):
        Sequence configuration.
    ModelConfig(BaseModel):
        Aggregates all model configuration settings.
"""

from pydantic import BaseModel, confloat, conint, model_validator

from components.assertions.choice_field_assertor import assert_choice_field
from pipeline.const import (
    MODEL_OPTIMIZATION_QUANTIZATION_DTYPES,
    MODEL_OPTIMIZATION_QUANTIZATION_ENGINE_NAMES,
)


class ModelOptimizationsPruningConfig(BaseModel):
    """Model optimizations pruning configuration.

    Attributes:
        amount (float): Pruning amount to apply (in [0.0, 1.0]).
    """

    amount: confloat(ge=0.0, le=1.0)


class ModelOptimizationsQuantizationConfig(BaseModel):
    """Model optimizations quantization configuration.

    Attributes:
        dtype (str): Target data type for quantization.
        engine (str): Quantization engine to use.
    """

    dtype: str
    engine: str

    @model_validator(mode="after")
    def check_model_optimizations_quantization_dtype(
        self: "ModelOptimizationsQuantizationConfig",
    ) -> "ModelOptimizationsQuantizationConfig":
        """Check whether quantization dtype is valid or not.

        This function validates the quantization dtype.

        Args:
            self (ModelOptimizationsQuantizationConfig): Current model instance.

        Returns:
            "ModelOptimizationsQuantizationConfig": Validated model instance.
        """
        assert_choice_field(
            self.dtype,
            MODEL_OPTIMIZATION_QUANTIZATION_DTYPES,
            "model.optimizations.quantization.dtype",
        )

        return self

    @model_validator(mode="after")
    def check_model_optimizations_quantization_engine(
        self: "ModelOptimizationsQuantizationConfig",
    ) -> "ModelOptimizationsQuantizationConfig":
        """Check whether quantization engine is valid or not.

        This function validates the quantization engine.

        Args:
            self (ModelOptimizationsQuantizationConfig): Current model instance.

        Returns:
            "ModelOptimizationsQuantizationConfig": Validated model instance.
        """
        assert_choice_field(
            self.engine,
            MODEL_OPTIMIZATION_QUANTIZATION_ENGINE_NAMES,
            "model.optimizations.quantization.engine",
        )

        return self


class ModelOptimizationsConfig(BaseModel):
    """Model optimizations configuration.

    Attributes:
        pruning (ModelOptimizationsPruningConfig): Pruning configuration.
        quantization (ModelOptimizationsQuantizationConfig): Quantization
                                                             configuration.
    """

    pruning: ModelOptimizationsPruningConfig
    quantization: ModelOptimizationsQuantizationConfig


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
        optimizations (ModelOptimizationsConfig): Model optimizations configuration.
    """

    params: ModelParamsConfig
    sequence: ModelSequenceConfig
    optimizations: ModelOptimizationsConfig
