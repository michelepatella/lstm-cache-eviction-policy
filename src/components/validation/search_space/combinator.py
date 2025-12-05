"""combinator.py

Module for generating all hyperparameter combinations from a configuration.

This module provides the `get_parameters_combination` function, which
processes a configuration search space, generates all possible combinations
of hyperparameters, and returns them as a list of fully nested dictionaries.
It supports complex nested configurations and ensures that every combination
from the search space is represented.

Functions:
    get_parameters_combination(pipeline_config: PipelineConfig) -> list[dict[str, int | float | bool]]
        Generates all nested parameter combinations from the provided search
        space configuration and returns them as a list of dictionaries.
"""

from components.dict.combinations.nested_dict_lists_combinator import (
    combine_nested_dict_lists,
)
from components.dict.combinations.nested_dicts_combinator import (
    combine_nested_dicts,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from pipeline.config.pydantic.pipeline_config import PipelineConfig


def get_parameters_combination(
    pipeline_config: PipelineConfig,
) -> list[dict[str, int | float | bool]]:
    """Generate all possible parameter combinations from the search space.

    This function flattens each section of the search space, generates
    all possible combinations of parameters, reconstructs the nested
    dictionary structure for each combination, and returns the list of
    fully nested parameter configurations.

    Args:
        pipeline_config (PipelineConfig): Configuration object.

    Returns:
        list[dict[str, int | float | bool]]: List of nested dictionaries
                                             representing all parameter
                                             combinations.

    Raises:
        RuntimeError: If parameter combinations generation fails:
            * The configuration object does not contain a valid search space
              or fails during access (AttributeError).
    """
    try:
        # Prepare configuration
        search_space = pipeline_config.validation.search_space.model_dump()

        debug(
            "Parameter combinations generation started",
            extra={
                "search_space": search_space,
                "context": "Parameter combinations generation",
            },
        )

        # Generate combinations per search
        # space section and save them
        section_combination_dicts = []
        for _, params_dict in search_space.items():
            section_values = combine_nested_dicts(params_dict)
            section_combination_dicts.append(section_values)

        # Combine all sections to get all the
        # final parameter combinations
        param_combinations = combine_nested_dict_lists(
            section_combination_dicts,
        )

        debug(
            "Parameter combinations generation completed",
            extra={
                "param_combinations_num": len(param_combinations),
                "sections_processed": list(search_space.keys()),
                "context": "Parameter combinations generation",
            },
        )

        return param_combinations
    except AttributeError as e:
        msg = "Parameter combinations generation failed"
        error(
            msg,
            extra={
                "exception": str(e),
                "context": "Parameter combinations generation",
            },
        )
        raise RuntimeError(msg) from e
