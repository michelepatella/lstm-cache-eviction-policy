"""initializer.py

Module responsible for initializing and configuring the Ray distributed
computing framework.

This module ensures that Ray is set up according to the specified configuration
parameters.

Functions:
    initialize_ray(ray_config: dict[str, Any]) -> None
        Initializes the Ray environment using the provided configuration.
"""

import ray

from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def initialize_ray(num_cpus: int, num_gpus: int) -> None:
    """Initializes the Ray environment.

    This initializes the Ray environment using the provided parameters.

    Args:
        num_cpus (int): Number of CPUs to allocate for Ray tasks.
        num_gpus (int): Number of GPUs to allocate for Ray tasks.

    Returns:
        None

    Raises:
        RuntimeError: If Ray initialization fails:
            * Invalid type in configuration values (TypeError).
            * Invalid value in configuration (ValueError).
    """
    try:
        # Shut down Ray if already initialized
        if ray.is_initialized():
            ray.shutdown()

        # Initialize Ray according to its config file
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

        debug(
            "Ray initialization executed",
            extra={
                "cpus_num": num_cpus,
                "gpus_num": num_cpus,
                "context": "Ray initialization",
            },
        )
    except (TypeError, ValueError) as e:
        msg = "Ray initialization failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "cpus_num": num_cpus,
                "gpus_num": num_cpus,
                "context": "Ray initialization",
            },
        )
        raise RuntimeError(msg) from e
