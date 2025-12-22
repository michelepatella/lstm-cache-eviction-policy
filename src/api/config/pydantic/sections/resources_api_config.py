"""resources_api_config.py

Module defining Pydantic schemas for resources configuration used
specifically within the API and simulation components.

These schemas ensure that the specified resources devices are valid choices
defined in the project constants, maintaining configuration integrity and
type safety.

Classes:
    ResourcesDeviceAPIConfig(BaseModel):
        Configuration schema for a single resources device, including
        device type validation.

    ResourcesAPIConfig(BaseModel):
        Top-level schema for resources configurations specific to the API.
"""

from pydantic import BaseModel, model_validator

from components.assertions.choice_field_assertor import assert_choice_field
from const import RESOURCES_DEVICE_NAMES


class ResourcesDeviceAPIConfig(BaseModel):
    """Resources device configuration for API.

    Attributes:
        type (str): The device type name.
    """

    type: str

    @model_validator(mode="after")
    def check_device(
        self: "ResourcesDeviceAPIConfig",
    ) -> "ResourcesDeviceAPIConfig":
        """Check whether device type is valid or not.

        This function validates the device type specified.

        Args:
            self (ResourcesDeviceAPIConfig): Current model instance.

        Returns:
            "ResourcesDevice": Validated model instance.
        """
        assert_choice_field(
            self.type,
            RESOURCES_DEVICE_NAMES,
            "resources.device.type",
        )

        return self


class ResourcesAPIConfig(BaseModel):
    """Resources configuration for API.

    Attributes:
        device (ResourcesDeviceAPIConfig): Resources device configuration.
    """

    device: ResourcesDeviceAPIConfig
