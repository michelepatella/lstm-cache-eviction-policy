from pydantic import BaseModel

from api.config.pydantic.sections.hardware_api_config import HardwareAPIConfig
from api.config.pydantic.sections.kwargs_api_config import KwargsAPIConfig
from api.const import API_CONFIG_USER_API_KWARG_FIELD_NAME
from components.logs.levels.error_logger import error


class APIConfig(BaseModel):
    """Global API configuration.

    Attributes:
        hardware (HardwareAPIConfig): Hardware configuration for the API.
        kwargs (KwargsAPIConfig): API kwargs configuration.
    """

    hardware: HardwareAPIConfig
    kwargs: KwargsAPIConfig

    def merge_api_kwargs(
        self: "APIConfig",
        user_kwargs: dict[str, int | float | list[int] | str | bool],
    ) -> KwargsAPIConfig:
        """Merge default API kwargs with user-provided ones.

        This function merges default API kwargs with those provided
        by the user, ensuring both values are validated.

        Args:
            self (APIConfig): Current model instance.
            user_kwargs (dict[str, int | float | list[int] | str | bool]):
                User-provided kwargs.

        Returns:
            KwargsAPIConfig: Merged API kwargs.
        """
        try:
            # For each API kwargs provided by the
            # user, merge its value with the default
            # one, giving precedence to the user-provided
            # value, provided that it is valid
            merged = self.kwargs.model_dump()
            for key, value in user_kwargs.items():
                if key in merged and value is not None:
                    merged[key] = {
                        **merged[key],
                        API_CONFIG_USER_API_KWARG_FIELD_NAME: value,
                    }

            return KwargsAPIConfig(**merged)
        except (KeyError, TypeError, ValueError) as e:
            msg = "API kwargs merging failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "user_api_kwargs": user_kwargs,
                    "api_kwargs": self.kwargs.model_dump(),
                    "context": "API kwargs merging",
                },
            )
            raise RuntimeError(msg) from e
