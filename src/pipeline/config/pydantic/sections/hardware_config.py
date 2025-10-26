from pydantic import BaseModel, model_validator

from components.assertions.choice_field_assertor import (
    assert_choice_field,
)
from pipeline.const import HW_DEVICE_NAMES


class HardwareConfig(BaseModel):
    """Hardware configuration.

    Attributes:
        device (str): Hardware device type to be used.
    """

    device: str

    @model_validator(mode="after")
    def check_device(self: "HardwareConfig") -> "HardwareConfig":
        """Check whether device is valid or not.

        This function validates the device specified.

        Args:
            self (HardwareConfig): Current model instance.

        Returns:
            "HardwareConfig": Validated model instance.
        """
        assert_choice_field(
            self.device,
            HW_DEVICE_NAMES,
            "hardware.device",
        )

        return self
