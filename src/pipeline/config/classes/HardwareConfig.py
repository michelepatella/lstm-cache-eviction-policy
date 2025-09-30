from pydantic import BaseModel, model_validator

from const import DEVICES
from pipeline.config.classes.validation.choice_field_validator import (
    validate_choice_field,
)


class HardwareConfig(BaseModel):
    """
    Class representing the hardware configuration
    settings.
    """

    device: str

    @model_validator(mode="after")
    def check_device(self: "HardwareConfig") -> "HardwareConfig":
        """
        Check whether device is valid or not.

        This function validates the device specified.

        Parameters:
            self (HardwareConfig): Current model instance.

        Returns:
            HardwareConfig: Validated model instance.
        """
        return validate_choice_field(
            self,
            self.device,
            DEVICES,
            "hardware.device",
        )
