from pydantic import BaseModel, model_validator

from components.assertions.choice_field_assertor import (
    assert_choice_field,
)
from const import HW_DEVICE_NAMES


class HardwareDeviceConfig(BaseModel):
    """Hardware device configuration.

        Attributes:
            type (str): Hardware device type to be used.
        """
    type: str

    @model_validator(mode="after")
    def check_device(self: "HardwareDeviceConfig") -> "HardwareDeviceConfig":
        """Check whether device type is valid or not.

        This function validates the device type specified.

        Args:
            self (HardwareDeviceConfig): Current model instance.

        Returns:
            "HardwareDeviceConfig": Validated model instance.
        """
        assert_choice_field(
            self.type,
            HW_DEVICE_NAMES,
            "hardware.device.type",
        )

        return self


class HardwareConfig(BaseModel):
    """Hardware configuration.

    Attributes:
        device (str): Hardware device configuration.
    """

    device: HardwareDeviceConfig