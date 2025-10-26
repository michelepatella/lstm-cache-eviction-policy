import numpy as np

from components.data.requests.utils.generation_helper import (
    generate_requests_helper,
)
from components.logs.levels.error_logger import error
from components.logs.levels.info_logger import info
from pipeline.config.pydantic.config import Config


def generate_dynamic_requests(
    config: Config,
) -> tuple[list[int], np.ndarray]:
    """Generate dynamic requests and corresponding timestamps in hours.

    This function generates dynamic requests and corresponding
    timestamps in hours. Dynamic requests change over time: multiple
    alpha values are generated between min and max, and total requests
    are split into time steps. Each time step uses a different alpha,
    creating temporal variability in the access distribution.

    Args:
        config (Config): Configuration object.

    Returns:
        tuple[list[int], np.ndarray]:
            - requests: List of generated keys requested.
            - timestamps_hours: Corresponding timestamps of requests in hours.

    Raises:
        RuntimeError: If generating dynamic requests fails:
            * Generating alpha values due to invalid min, max, or step values
              (ValueError, TypeError).
            * Converting alpha values to list due to invalid sequence
              (TypeError).
    """
    try:
        # Prepare configuration
        zipf_config = config.data.pattern.access.zipf
        alpha_min = zipf_config.alpha.min
        alpha_max = zipf_config.alpha.max
        steps = zipf_config.steps
        num_requests = config.data.requests

        # Generate evenly spaced alpha
        # values for dynamic time steps
        alpha_range = np.linspace(alpha_min, alpha_max, steps).tolist()

        # Calculate time step duration for
        # requests generation
        time_step_duration = num_requests // len(alpha_range)

        info(
            "Dynamic requests generation started",
            extra={
                "alpha_min": alpha_min,
                "alpha_max": alpha_max,
                "steps": steps,
                "requests_num": num_requests,
                "time_step_duration": time_step_duration,
                "alpha_range": alpha_range,
                "context": "Dynamic requests generation",
            },
        )

        # Use common helper to generate
        # requests based on dynamic alpha range
        requests, timestamps_hours = generate_requests_helper(
            alpha_range,
            config,
            time_step_duration,
        )

        info(
            "Dynamic requests generation completed",
            extra={
                "requests_generated_num": len(requests),
                "timestamps_generated_num": len(timestamps_hours),
                "context": "Dynamic requests generation",
            },
        )

        return requests, timestamps_hours
    except (ValueError, TypeError) as e:
        msg = "Dynamic requests generation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "alpha_min": alpha_min,
                "alpha_max": alpha_max,
                "steps": steps,
                "requests_num": num_requests,
                "context": "Dynamic requests generation",
            },
        )
        raise RuntimeError(msg) from e
