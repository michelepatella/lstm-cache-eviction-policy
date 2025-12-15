"""model_pipeline_config.py

Configuration section for the model's architecture,
sequence handling, and optimizations.

This module defines the core parameters of the model, sequence processing
(embedding, length), and post-training optimizations (pruning and quantization).

Classes:
    ModelOptimizationsPruningPipelineConfig(BaseModel):
        Pruning configuration.
    ModelOptimizationsQuantizationPipelineConfig(BaseModel):
        Quantization configuration.
    ModelOptimizationsPipelineConfig(BaseModel):
        Aggregates model optimization settings.
    ModelParamsPipelineConfig(BaseModel):
        Model parameters' configuration.
    ModelSequenceEmbeddingPipelineConfig(BaseModel):
        Embedding configuration.
    ModelSequencePipelineConfig(BaseModel):
        Sequence configuration.
    ModelPipelineConfig(BaseModel):
        Aggregates all model configuration settings.
"""

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from components.assertions.choice_field_assertor import assert_choice_field
from const import MODEL_OPTIMIZATIONS_QUANTIZATION_ENGINE_NAMES
from pipeline.const import (
    MODEL_OPTIMIZATIONS_QUANTIZATION_DTYPES,
)


class ModelOptimizationsPruningPipelineConfig(BaseModel):
    """Model optimizations pruning configuration.

    Attributes:
        amount (float): Pruning amount to apply (in [0.0, 1.0]).
    """

    amount: Annotated[float, Field(ge=0.0, le=1.0)]


class ModelOptimizationsQuantizationPipelineConfig(BaseModel):
    """Model optimizations quantization configuration.

    Attributes:
        dtype (str): Target data type for quantization.
        engine (str): Quantization engine to use.
    """

    dtype: str
    engine: str

    @model_validator(mode="after")
    def check_model_optimizations_quantization_dtype(
        self: "ModelOptimizationsQuantizationPipelineConfig",
    ) -> "ModelOptimizationsQuantizationPipelineConfig":
        """Check whether quantization dtype is valid or not.

        This function validates the quantization dtype.

        Args:
            self (ModelOptimizationsQuantizationPipelineConfig): Current model instance.

        Returns:
            "ModelOptimizationsQuantizationConfig": Validated model instance.
        """
        assert_choice_field(
            self.dtype,
            MODEL_OPTIMIZATIONS_QUANTIZATION_DTYPES,
            "model.optimizations.quantization.dtype",
        )

        return self

    @model_validator(mode="after")
    def check_model_optimizations_quantization_engine(
        self: "ModelOptimizationsQuantizationPipelineConfig",
    ) -> "ModelOptimizationsQuantizationPipelineConfig":
        """Check whether quantization engine is valid or not.

        This function validates the quantization engine.

        Args:
            self (ModelOptimizationsQuantizationPipelineConfig): Current model instance.

        Returns:
            "ModelOptimizationsQuantizationConfig": Validated model instance.
        """
        assert_choice_field(
            self.engine,
            MODEL_OPTIMIZATIONS_QUANTIZATION_ENGINE_NAMES,
            "model.optimizations.quantization.engine",
        )

        return self


class ModelOptimizationsPipelineConfig(BaseModel):
    """Model optimizations configuration.

    Attributes:
        pruning (ModelOptimizationsPruningPipelineConfig): Pruning configuration.
        quantization (ModelOptimizationsQuantizationPipelineConfig):
            Quantization configuration.
    """

    pruning: ModelOptimizationsPruningPipelineConfig
    quantization: ModelOptimizationsQuantizationPipelineConfig


class ModelParamsPipelineConfig(BaseModel):
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

    hidden_size: Annotated[int, Field(gt=0)]
    num_layers: Annotated[int, Field(gt=0)]
    bias: bool
    batch_first: bool
    dropout: Annotated[float, Field(ge=0, lt=1)]
    bidirectional: bool
    proj_size: Annotated[int, Field(ge=0)]


class ModelSequenceEmbeddingPipelineConfig(BaseModel):
    """Embedding configuration for sequence input.

    Attributes:
        dimension (int): Dimension of the embedding vector (> 0).
    """

    dimension: Annotated[int, Field(gt=0)]


class ModelSequencePipelineConfig(BaseModel):
    """Sequence configuration for the model.

    Attributes:
        length (int): Length of the input sequences (> 0).
        embedding (ModelSequenceEmbeddingPipelineConfig): Embedding configuration.
    """

    length: Annotated[int, Field(gt=0)]
    embedding: ModelSequenceEmbeddingPipelineConfig


class ModelPipelineConfig(BaseModel):
    """Model configuration.

    Attributes:
        params (ModelParamsPipelineConfig): Model layer configuration.
        sequence (ModelSequencePipelineConfig): Sequence and embedding
                                                configuration.
        optimizations (ModelOptimizationsPipelineConfig): Model optimizations
                                                          configuration.
    """

    params: ModelParamsPipelineConfig
    sequence: ModelSequencePipelineConfig
    optimizations: ModelOptimizationsPipelineConfig
