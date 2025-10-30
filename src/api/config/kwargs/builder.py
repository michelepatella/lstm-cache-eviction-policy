from fastapi import HTTPException, status

from api.config.kwargs.APIKwargs import APIKwargs
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def build_api_kwargs(
    default_api_kwargs: dict[str, int | float | list[int] | str | bool],
    user_api_kwargs: dict[str, int | float | list[int] | str | bool] | None,
) -> APIKwargs:
    """Build API kwargs starting from default and user-provided ones.

    This function combines the default API kwargs with any user-supplied
    values, giving precedence to user values. If no user API kwargs are
    provided or a specific key is missing, the default value is used.
    Resulting API kwarg values are validated before being returned.

    Args:
        default_api_kwargs (dict[str, int | float | list[int] | str | bool]):
            Default API kwargs.
        user_api_kwargs (dict[str, int | float | list[int] | str | bool] | None):
            User-provided API kwargs (None if any has been provided).

    Returns:
        APIKwargs: Instance containing the final API kwargs.

    Raises:
        HTTPException: If API kwargs building fails:
            * If default API kwargs are missing or malformed (KeyError).
            * If merging with user API kwargs fails due to
              invalid value (ValueError).
            * If merging with user API kwargs fails due to type
              or structure issues (TypeError).
    """
    try:
        debug(
            "API kwargs building started",
            extra={
                "api_kwargs_default": list(default_api_kwargs.keys()),
                "api_kwargs_user": list(user_api_kwargs.keys())
                if user_api_kwargs
                else None,
                "context": "API kwargs building",
            },
        )

        # If the user provided API kwargs, merge
        # them with default ones, checking their
        # validity before
        if user_api_kwargs:
            merged_api_kwargs = {**default_api_kwargs, **user_api_kwargs}

        # Otherwise, use default API kwarg values
        else:
            merged_api_kwargs = default_api_kwargs

        # Instantiate API kwargs object
        api_kwargs = APIKwargs(**merged_api_kwargs)

        debug(
            "API kwargs building completed",
            extra={
                "api_kwargs": list(merged_api_kwargs.keys()),
                "context": "API kwargs building",
            },
        )

        return api_kwargs
    except (TypeError, ValueError, KeyError) as e:
        error(
            "API kwargs building failed",
            extra={
                "exception": str(e),
                "api_kwargs_default": list(default_api_kwargs.keys()),
                "api_kwargs_user": list(user_api_kwargs.keys())
                if user_api_kwargs
                else None,
                "context": "API kwargs building",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
