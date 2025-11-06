"""caches_config.py

Configuration section for general cache parameters used during simulations.

This module defines core parameters common to most cache implementations,
such as the cache size and the Time-To-Live (TTL) setting.

Classes:
    CachesConfig(BaseModel):
        Configuration for general cache settings.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class CachesConfig(BaseModel):
    """Configuration for general cache settings.

    Attributes:
        dimension (int): The maximum size (number of keys)
                            of the cache (>= 1).
        ttl (int): Time-To-Live for cache entries (>= 0).
    """

    dimension: Annotated[int, Field(ge=1)]
    ttl: Annotated[int, Field(ge=0)]
