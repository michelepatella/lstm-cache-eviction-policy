from pydantic import BaseModel, model_validator

from const import HW_DEVICES
from utils.assertions.choice_field_validator import (
    validate_choice_field,
)


class HardwareConfig(BaseModel):
    """
    Hardware configuration.

    Attributes:
        device (str): Hardware device type to be used.
    """

    device: str

    @model_validator(mode="after")
    def check_device(self: "HardwareConfig") -> "HardwareConfig":
        """
        Check whether device is valid or not.

        This function validates the device specified.

        Args:
            self (HardwareConfig): Current model instance.

        Returns:
            HardwareConfig: Validated model instance.
        """
        return validate_choice_field(
            self,
            self.device,
            HW_DEVICES,
            "hardware.device",
        )
