"""model_api_config.py

Module defining Pydantic schemas for configuring the model.

These schemas ensure that the chosen model optimization parameters,
such as the quantization engine, are valid choices defined in the
project constants. This guarantees both type safety and operational
integrity when deploying the model.

Classes:
    ModelOptimizationsQuantizationAPIConfig(BaseModel):
        Configuration and validation schema for the model's quantization
        engine.

    ModelOptimizationsAPIConfig(BaseModel):
        Top-level schema for model optimization settings.

    ModelAPIConfig(BaseModel):
        The main, top-level configuration schema for the model parameters
        used by the API.
"""

from pydantic import BaseModel, model_validator

from components.assertions.choice_field_assertor import assert_choice_field
from const import MODEL_OPTIMIZATIONS_QUANTIZATION_ENGINE_NAMES


class ModelOptimizationsQuantizationAPIConfig(BaseModel):
    """Model optimizations quantization configuration.

    Attributes:
        engine (str): Quantization engine to use.
    """

    engine: str

    @model_validator(mode="after")
    def check_model_optimizations_quantization_engine(
        self: "ModelOptimizationsQuantizationAPIConfig",
    ) -> "ModelOptimizationsQuantizationAPIConfig":
        """Check whether quantization engine is valid or not.

        This function validates the quantization engine.

        Args:
            self (ModelOptimizationsQuantizationAPIConfig): Current model instance.

        Returns:
            "ModelOptimizationsQuantizationConfig": Validated model instance.
        """
        assert_choice_field(
            self.engine,
            MODEL_OPTIMIZATIONS_QUANTIZATION_ENGINE_NAMES,
            "model.optimizations.quantization.engine",
        )

        return self


class ModelOptimizationsAPIConfig(BaseModel):
    """Model optimizations configuration.

    Attributes:
        quantization (ModelOptimizationsQuantizationAPIConfig):
            Quantization configuration.
    """

    quantization: ModelOptimizationsQuantizationAPIConfig


class ModelAPIConfig(BaseModel):
    """Model configuration.

    Attributes:
        optimizations (ModelOptimizationsAPIConfig): Model optimizations
                                                     configuration.
    """

    optimizations: ModelOptimizationsAPIConfig
