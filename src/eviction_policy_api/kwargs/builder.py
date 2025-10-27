from fastapi import HTTPException, status

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from eviction_policy_api.kwargs.APIKwargs import APIKwargs


def build_api_kwargs(
    default_kwargs: dict[str, int | float | list[int] | str | bool],
    user_kwargs: dict[str, int | float | list[int] | str | bool] | None,
) -> APIKwargs:
    """Build API kwargs starting from default and user-provided ones.

    This function combines the default API kwargs with any user-supplied
    values, giving precedence to user values. If no user kwargs are
    provided or a specific key is missing, the default value is used.
    Resulting kwarg values are validated before being returned.

    Args:
        default_kwargs (dict[str, int | float | list[int] | str | bool]):
            Default API kwargs.
        user_kwargs (dict[str, int | float | list[int] | str | bool] | None):
            User-provided kwargs (None if any has been provided).

    Returns:
        APIKwargs: Instance containing the final API kwargs.

    Raises:
        HTTPException: If kwargs building fails:
            * If default kwargs are missing or malformed (KeyError).
            * If merging with user kwargs fails due to
              invalid value (ValueError).
            * If merging with user kwargs fails due to type
              or structure issues (TypeError).
    """
    try:
        debug(
            "API kwargs building started",
            extra={
                "api_kwargs_default": list(default_kwargs.keys()),
                "api_kwargs_user": list(user_kwargs.keys())
                if user_kwargs
                else None,
                "context": "API kwargs building",
            },
        )

        # If the user provided kwargs, merge
        # them with default ones, checking their
        # validity before
        if user_kwargs:
            merged_kwargs = {**default_kwargs, **user_kwargs}

        # Otherwise, use default kwarg values
        else:
            merged_kwargs = default_kwargs

        # Instantiate API kwargs object
        api_kwargs = APIKwargs(**merged_kwargs)

        debug(
            "API kwargs building completed",
            extra={
                "api_kwargs_final": list(merged_kwargs.keys()),
                "context": "API kwargs building",
            },
        )

        return api_kwargs
    except (TypeError, ValueError, KeyError) as e:
        error(
            "API kwargs building failed",
            extra={
                "exception": str(e),
                "api_kwargs_default": list(default_kwargs.keys()),
                "api_kwargs_user": list(user_kwargs.keys())
                if user_kwargs
                else None,
                "context": "API kwargs building",
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
