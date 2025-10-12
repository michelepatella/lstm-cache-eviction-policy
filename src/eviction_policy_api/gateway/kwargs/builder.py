from typing import Any, Dict, List

from fastapi import HTTPException, status

from eviction_policy_api.gateway.kwargs.utils.APIKwargs import APIKwargs


def build_api_kwargs(
    default_kwargs: Dict[str, int | float | List[Any] | str | bool],
    user_kwargs: Dict[str, int | float | List[Any] | str | bool] | None,
) -> APIKwargs:
    """
    Build API kwargs starting from default and user-provided ones.

    This function combines the default API kwargs with any user-supplied
    values, giving precedence to user values. If no user kwargs are
    provided or a specific key is missing, the default value is used.
    Resulting kwarg values are validated before being returned.

    Args:
        default_kwargs (Dict[str, int | float | List[Any] | str | bool]):
            Default API kwargs.
        user_kwargs (Dict[str, int | float | List[Any] | str | bool] | None):
            User-provided kwargs (None if any has been provided).

    Returns:
        APIKwargs: Instance containing the final API kwargs.

    Raises:
        HTTPException: If an error occurs during kwargs building, e.g.:
            * If default kwargs are missing or malformed.
            * If merging with user kwargs fails due to
              invalid value.
            * If merging with user kwargs fails due to type
              or structure issues.
    """
    try:
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

        return api_kwargs
    except TypeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid type or structure in API kwargs",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid value in API kwargs",
        ) from e
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Missing expected key in API kwargs",
        ) from e