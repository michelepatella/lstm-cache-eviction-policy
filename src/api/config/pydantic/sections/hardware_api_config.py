from pydantic import BaseModel, model_validator

from components.assertions.choice_field_assertor import assert_choice_field
from const import HW_DEVICE_NAMES


class HardwareDeviceAPIConfig(BaseModel):
    """Hardware device configuration for API.

    Attributes:
        type (str): The device type name.
    """

    type: str

    @model_validator(mode="after")
    def check_device(
        self: "HardwareDeviceAPIConfig",
    ) -> "HardwareDeviceAPIConfig":
        """Check whether device type is valid or not.

        This function validates the device type specified.

        Args:
            self (HardwareDeviceAPIConfig): Current model instance.

        Returns:
            "HardwareDevice": Validated model instance.
        """
        assert_choice_field(
            self.type,
            HW_DEVICE_NAMES,
            "hardware.device.type",
        )

        return self


class HardwareAPIConfig(BaseModel):
    """Hardware configuration for API.

    Attributes:
        device (HardwareDeviceAPIConfig): Hardware device configuration.
    """

    device: HardwareDeviceAPIConfig
