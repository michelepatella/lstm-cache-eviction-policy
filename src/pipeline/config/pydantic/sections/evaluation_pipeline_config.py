"""evaluation_pipeline_config.py

Configuration section for defining evaluation procedures and metrics.

This module structures parameters related to specific metrics used during
simulations and testing.

Classes:
    EvaluationSimulationsMetricsMistakeRatePipelineConfig(BaseModel):
        Configuration for mistake rate calculation.
    EvaluationSimulationsMetricsPipelineConfig(BaseModel):
        Configuration for all simulation metrics.
    EvaluationSimulationsPipelineConfig(BaseModel):
        Configuration for simulation evaluation.
    EvaluationPipelineConfig(BaseModel):
        Aggregates all evaluation-related settings.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class EvaluationSimulationsMetricsMistakeRatePipelineConfig(BaseModel):
    """Configuration for mistake rate calculation.

    Attributes:
        window (int): The look-ahead window size used to
                      determine an eviction mistake (>= 1).
    """

    window: Annotated[int, Field(ge=1)]


class EvaluationSimulationsMetricsPipelineConfig(BaseModel):
    """Configuration for all simulation metrics.

    Attributes:
        mistake_rate (EvaluationSimulationsMetricsMistakeRatePipelineConfig):
            Mistake rate configuration.
    """

    mistake_rate: EvaluationSimulationsMetricsMistakeRatePipelineConfig


class EvaluationSimulationsPipelineConfig(BaseModel):
    """Configuration for simulation evaluation.

    Attributes:
        metrics (EvaluationSimulationsMetricsPipelineConfig): Metrics calculation
                                                              configuration.
    """

    metrics: EvaluationSimulationsMetricsPipelineConfig


class EvaluationPipelineConfig(BaseModel):
    """Aggregates all evaluation-related settings.

    Attributes:
        simulations (EvaluationSimulationsPipelineConfig): Simulation evaluation
                                                           configuration.
    """

    simulations: EvaluationSimulationsPipelineConfig
