"""initializer.py

Module responsible for initializing and configuring the Ray distributed
computing framework.

This module ensures that Ray is set up according to the specified configuration
parameters.

Functions:
    initialize_ray(ray_config: dict[str, Any]) -> None
        Initializes the Ray environment using the provided configuration.
"""

from typing import Any

import ray
from box import Box

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def initialize_ray(ray_config: dict[str, Any]) -> None:
    """Initializes the Ray environment.

    This initializes the Ray environment using the provided configuration
    parameters.

    Args:
        ray_config (dict[str, Any]): A dictionary containing Ray initialization
                                     parameters.

    Returns:
        None

    Raises:
        RuntimeError: If Ray initialization fails:
            * Invalid type in configuration values (TypeError).
            * Invalid value in configuration (ValueError).
            * Missing expected configuration attributes (AttributeError).
    """
    try:
        # Box Ray configuration
        ray_config = Box(ray_config)

        # Shut down Ray if already initialized
        if ray.is_initialized():
            ray.shutdown()

        # Initialize Ray according to its config file
        ray.init(
            ignore_reinit_error=ray_config.ignore_reinit_error,
            include_dashboard=ray_config.include_dashboard,
            log_to_driver=ray_config.log_to_driver,
            local_mode=ray_config.local_mode,
            num_cpus=ray_config.num_cpus,
            num_gpus=ray_config.num_gpus,
            object_store_memory=ray_config.object_store_memory,
        )

        debug(
            "Ray initialization executed",
            extra={"ray_config": ray_config, "context": "Ray initialization"},
        )
    except (TypeError, ValueError, AttributeError) as e:
        msg = "Ray initialization failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "ray_config": ray_config,
                "context": "Ray initialization",
            },
        )
        raise RuntimeError(msg) from e
