"""loss_pipeline_config.py

Configuration section for the loss function settings.

This module defines parameters related to the loss function.

Classes:
    LossClassWeightPipelineConfig(BaseModel):
        Configuration for class weight calculation.
    LossPipelineConfig(BaseModel):
        Aggregates all loss-related settings.
"""

from pydantic import BaseModel, model_validator

from components.assertions.choice_field_assertor import assert_choice_field
from pipeline.const import LOSS_CLASS_WEIGHT_TYPES


class LossClassWeightPipelineConfig(BaseModel):
    """Configuration for class weight calculation.

    Attributes:
        type (str): Type of class weighting to use.
    """

    type: str

    @model_validator(mode="after")
    def check_loss_class_weight_type(
        self: "LossClassWeightPipelineConfig",
    ) -> "LossClassWeightPipelineConfig":
        """Check whether loss class weight type is valid or not.

        This function validates the loss class weight type.

        Args:
            self (LossClassWeightPipelineConfig): Current model instance.

        Returns:
            "LossClassWeightConfig": Validated model instance.
        """
        assert_choice_field(
            self.type,
            LOSS_CLASS_WEIGHT_TYPES,
            "loss.class_weight.type",
        )

        return self


class LossPipelineConfig(BaseModel):
    """Aggregates all loss-related settings.

    Attributes:
        class_weight (LossClassWeightPipelineConfig): Configuration for class
                                                      weight calculation.
    """

    class_weight: LossClassWeightPipelineConfig
