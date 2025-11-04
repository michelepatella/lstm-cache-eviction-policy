"""caches_config.py

Configuration section for general cache parameters used during simulations.

This module defines core parameters common to most cache implementations,
such as the cache size and the Time-To-Live (TTL) setting.

Classes:
    CachesConfig: Configuration for general cache settings.
"""

from pydantic import BaseModel, conint


class CachesConfig(BaseModel):
    """Configuration for general cache settings.

    Attributes:
        dimension (conint): The maximum size (number of keys)
                            of the cache (>= 1).
        ttl (conint): Time-To-Live for cache entries (>= 0).
    """

    dimension: conint(ge=1)
    ttl: conint(ge=0)
