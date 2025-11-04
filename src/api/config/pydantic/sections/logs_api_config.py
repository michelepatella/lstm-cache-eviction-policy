"""logs_api_config.py

Configuration section for logging settings within the API.

This module defines the desired logging level for the API.

Classes:
    LogsConfig(BaseModel):
        Configuration for the API logging system.
"""

from pydantic import BaseModel, model_validator

from components.assertions.choice_field_assertor import assert_choice_field
from pipeline.const import LOGS_LEVEL_NAMES


class LogsAPIConfig(BaseModel):
    """Configuration for API logging.

    Attributes:
        level (str): The minimum logging level to output.
    """

    level: str

    @model_validator(mode="after")
    def check_logs_level(
        self: "LogsAPIConfig",
    ) -> "LogsAPIConfig":
        """Check whether logs level is valid or not.

        This function validates the logs level.

        Args:
            self (LogsAPIConfig): Current model instance.

        Returns:
            "LogsConfig": Validated model instance.
        """
        assert_choice_field(
            self.level,
            LOGS_LEVEL_NAMES,
            "logs.level",
        )

        return self
