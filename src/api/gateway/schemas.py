"""schemas.py

This module defines the Pydantic model used for validating and structuring
the request body (input data) sent to the central API Gateway endpoint
responsible for the cache eviction policy.

Classes:
    GatewayAPIInput(BaseModel):
        Defines the required and optional input fields for the cache eviction
        pipeline, ensuring data integrity before processing.
"""

from pydantic import BaseModel


class GatewayAPIInput(BaseModel):
    """Pydantic model for validating input data to the Gateway API.

    This model strictly defines the structure and types for the input parameters
    required to run the cache eviction pipeline.

    Attributes:
        keys_in_cache (List[int]):
            A list of unique integer keys currently residing in the cache.
        last_accesses (List[Tuple[float, int]]):
            A chronological sequence of recent cache accesses. Each tuple must
            contain (float: time_of_day_in_hours, int: key_accessed).
        user_api_kwargs (Optional[Dict[str, Union[int, float, str, bool, List[int]]]]):
            An optional dictionary used to override default configuration
            parameters (kwargs) for the cache eviction pipeline at runtime.
            Keys are configuration names (str), and values can be primitive
            types or list of integers. Defaults to None.
    """

    keys_in_cache: list[int]
    last_accesses: list[tuple[float, int]]
    user_api_kwargs: dict[str, int | float | str | bool | list[int]] | None = (
        None
    )
