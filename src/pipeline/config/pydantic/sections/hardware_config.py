from pydantic import BaseModel, model_validator

from components.assertions.choice_field_assertor import (
    assert_choice_field,
)
from const import HW_DEVICE_NAMES


class HardwareConfig(BaseModel):
    """Hardware configuration.

    Attributes:
        device_type (str): Hardware device type to be used.
    """

    device_type: str

    @model_validator(mode="after")
    def check_device(self: "HardwareConfig") -> "HardwareConfig":
        """Check whether device type is valid or not.

        This function validates the device type specified.

        Args:
            self (HardwareConfig): Current model instance.

        Returns:
            "HardwareConfig": Validated model instance.
        """
        assert_choice_field(
            self.device_type,
            HW_DEVICE_NAMES,
            "hardware.device_type",
        )

        return self
