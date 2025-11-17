"""resources_pipeline_config.py

Configuration section for hardware resource allocation.

This module structures parameters defining the availability of hardware
resources and which devices are used for specific pipeline phases
(training, testing, validation).

Classes:
    ResourcesGeneralPipelineConfig(BaseModel):
        Defines the total number of CPUs and GPUs available.
    ResourcesDevicesPipelineConfig(BaseModel):
        Specifies the device type to use for each phase.
    ResourcesPipelineConfig(BaseModel):
        Aggregates all resource configuration settings.
"""

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from components.assertions.choice_field_assertor import (
    assert_choice_field,
)
from src.const import RESOURCES_DEVICE_NAMES


class ResourcesGeneralPipelineConfig(BaseModel):
    """General hardware resource allocation.

    Attributes:
        num_cpus (int): Total number of CPU cores available (>= 1).
        num_gpus (int): Total number of GPUs available (>= 0).
    """

    num_cpus: Annotated[int, Field(ge=1)]
    num_gpus: Annotated[int, Field(ge=0)]


class ResourcesDevicesPipelineConfig(BaseModel):
    """Device selection for each pipeline phase.

    Attributes:
        training (str): Device type for the training phase.
        testing (str): Device type for the testing phase.
        validation (str): Device type for the validation phase.
    """

    training: str
    testing: str
    validation: str

    @model_validator(mode="after")
    def check_device_types(
        self: "ResourcesDevicesPipelineConfig",
    ) -> "ResourcesDevicesPipelineConfig":
        """Check whether all device types are valid.

        Args:
            self (ResourcesDevicesPipelineConfig): Current model instance.

        Returns:
            "ResourcesDevicesConfig": Validated model instance.
        """
        assert_choice_field(
            self.training,
            RESOURCES_DEVICE_NAMES,
            "resources.devices.training",
        )
        assert_choice_field(
            self.testing,
            RESOURCES_DEVICE_NAMES,
            "resources.devices.testing",
        )
        assert_choice_field(
            self.validation,
            RESOURCES_DEVICE_NAMES,
            "resources.devices.validation",
        )
        return self


class ResourcesPipelineConfig(BaseModel):
    """Resources configuration.

    Attributes:
        general (ResourcesGeneralPipelineConfig): General hardware availability.
        devices (ResourcesDevicesPipelineConfig): Device selection per pipeline
                                                  phase.
    """

    general: ResourcesGeneralPipelineConfig
    devices: ResourcesDevicesPipelineConfig
