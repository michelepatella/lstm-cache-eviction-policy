"""testing_config.py

Configuration section for defining the testing environment and procedures.

This module structures parameters related to running the model evaluation
on the testing dataset, including hardware selection and general test
parameters.

Classes:
    TestingDeviceConfig(BaseModel):
        Configuration for the hardware device used during testing.
    TestingGeneralConfig(BaseModel):
        General configuration parameters for the test run.
    TestingConfig(BaseModel):
        Aggregates all testing configuration settings.
"""

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from components.assertions.choice_field_assertor import assert_choice_field
from const import HW_DEVICE_NAMES


class TestingDeviceConfig(BaseModel):
    """Device configuration for testing.

    Attributes:
        type (str): The device type to use.
    """

    type: str

    @model_validator(mode="after")
    def check_testing_device_type(
        self: "TestingDeviceConfig",
    ) -> "TestingDeviceConfig":
        """Check whether testing device type is valid or not.

        This function validates the testing device type.

        Args:
            self (TestingDeviceConfig): Current model instance.

        Returns:
            "TestingDeviceConfig": Validated model instance.
        """
        assert_choice_field(
            self.type,
            HW_DEVICE_NAMES,
            "testing.device.type",
        )

        return self


class TestingGeneralConfig(BaseModel):
    """General configuration for testing.

    Attributes:
        batch_size (int): Number of samples per batch (> 0).
        shuffle (bool): Whether to shuffle the dataset during testing.
    """

    batch_size: Annotated[int, Field(gt=0)]
    shuffle: bool


class TestingConfig(BaseModel):
    """Testing configuration.

    Attributes:
        device (TestingDeviceConfig): Device configuration for testing.
        general (TestingGeneralConfig): General testing configuration.
    """

    device: TestingDeviceConfig
    general: TestingGeneralConfig
