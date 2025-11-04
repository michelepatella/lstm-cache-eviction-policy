"""evaluation_config.py

Configuration section for defining evaluation procedures and metrics.

This module structures parameters related to specific metrics used during
simulations and testing.

Classes:
    EvaluationSimulationsMetricsMistakeRateConfig: Configuration for mistake
                                                  rate calculation.
    EvaluationSimulationsMetricsConfig: Configuration for all simulation metrics.
    EvaluationSimulationsConfig: Configuration for simulation evaluation.
    EvaluationConfig: Aggregates all evaluation-related settings.
"""

from pydantic import BaseModel, conint


class EvaluationSimulationsMetricsMistakeRateConfig(BaseModel):
    """Configuration for mistake rate calculation.

    Attributes:
        window (conint): The look-ahead window size used to
                         determine an eviction mistake (>= 1).
    """

    window: conint(ge=1)


class EvaluationSimulationsMetricsConfig(BaseModel):
    """Configuration for all simulation metrics.

    Attributes:
        mistake_rate (EvaluationSimulationsMetricsMistakeRateConfig):
            Mistake rate configuration.
    """

    mistake_rate: EvaluationSimulationsMetricsMistakeRateConfig


class EvaluationSimulationsConfig(BaseModel):
    """Configuration for simulation evaluation.

    Attributes:
        metrics (EvaluationSimulationsMetricsConfig):
            Metrics calculation configuration.
    """

    metrics: EvaluationSimulationsMetricsConfig


class EvaluationConfig(BaseModel):
    """Aggregates all evaluation-related settings.

    Attributes:
        simulations (EvaluationSimulationsConfig):
            Simulation evaluation configuration.
    """

    simulations: EvaluationSimulationsConfig
