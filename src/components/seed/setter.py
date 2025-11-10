"""setter.py

Module dedicated to ensuring the reproducibility of results across all
randomized components and libraries used in the project.

Reproducibility is achieved by setting a fixed seed value for the Python
built-in random module, NumPy, and PyTorch, including specific settings
for PyTorch's CUDA backend (cuDNN) to guarantee deterministic behavior
during operations that might otherwise introduce non-determinism.

Functions:
    set_seed(seed: int) -> None
        Sets the global random seed for necessary libraries and
        configures PyTorch backends.
"""

import random, numpy as np, torch

from components.const import (
    TORCH_BACKENDS_CUDNN_BENCHMARK,
    TORCH_BACKENDS_CUDNN_DETERMINISTIC,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error


def set_seed(seed: int) -> None:
    """Sets the random seed for Python's standard random
    library, NumPy, and PyTorch.

    This function ensures that operations involving randomness
    produce the same sequence of results across different executions, by
    setting a global seed.

    Args:
        seed (int): The integer seed value to be set.

    Returns:
        None

    Raises:
        RuntimeError: If seed setting fails:
            * Invalid type (TypeError).
            * Invalid value (ValueError).
    """
    try:
        # Set a seed for all the libraries
        # and frameworks used across the
        # project to ensure reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = TORCH_BACKENDS_CUDNN_DETERMINISTIC
        torch.backends.cudnn.benchmark = TORCH_BACKENDS_CUDNN_BENCHMARK

        debug(
            "Seed setting executed",
            extra={
                "seed": seed,
                "context": "Seed setting",
            },
        )
    except (TypeError, ValueError) as e:
        msg = "Seed setting failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "seed": seed,
                "context": "Seed setting",
            },
        )
        raise RuntimeError(msg) from e
