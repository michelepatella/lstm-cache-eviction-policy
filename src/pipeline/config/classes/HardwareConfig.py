from pydantic import BaseModel, model_validator

from pipeline.const import HW_DEVICES
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
    def check_device(self: "HardwareConfig") -> None:
        """
        Check whether device is valid or not.

        This function validates the device specified.

        Args:
            self (HardwareConfig): Current model instance.

        Returns:
            None.
        """
        validate_choice_field(
            self.device,
            HW_DEVICES,
            "hardware.device",
        )
