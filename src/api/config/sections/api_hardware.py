from pydantic import BaseModel, model_validator

from components.assertions.choice_field_assertor import assert_choice_field
from const import HW_DEVICE_NAMES


class APIHardwareDevice(BaseModel):
    """Hardware device configuration for API.

    This class defines the hardware device type used
    for running the API.

    Attributes:
        type (str): The device type name.
    """

    type: str

    @model_validator(mode="after")
    def check_device(self: "APIHardwareDevice") -> "APIHardwareDevice":
        """Check whether device type is valid or not.

        This function validates the device type specified.

        Args:
            self (APIHardwareDevice): Current model instance.

        Returns:
            "HardwareDevice": Validated model instance.
        """
        assert_choice_field(
            self.type,
            HW_DEVICE_NAMES,
            "hardware.device.type",
        )

        return self


class APIHardware(BaseModel):
    """Hardware configuration for API.

    This class encapsulates hardware-related
    configuration for the API.

    Attributes:
        device (APIHardwareDevice): Hardware device configuration.
    """

    device: APIHardwareDevice
