"""data_pipeline_config.py

Configuration section for data parameters.

This module structures all parameters necessary for generating synthetic
request data, covering core aspects like key ranges, total request count,
distribution mode, and complex temporal and access patterns.

It defines parameters for Zipf distribution, burstiness, and various access
behaviors (repetition, toggle, cycle, distortion, and memory effects).

Classes:
    DataPatternsAccessBehaviorHoursPipelineConfig(BaseModel):
        Hour range configuration.
    DataKeysPipelineConfig(BaseModel):
        Configuration for key ID ranges.
    DataPatternsAccessZipfAlphaPipelineConfig(BaseModel):
        Alpha parameters for Zipf distribution.
    DataPatternsAccessZipfPipelineConfig(BaseModel):
        Configuration for Zipf distribution.
    DataPatternsAccessBehaviorRepetitionPipelineConfig(BaseModel):
        Repetition behavior configuration.
    DataPatternsAccessBehaviorToggleOffsetsPipelineConfig(BaseModel):
        Toggle offsets configuration.
    DataPatternsAccessBehaviorToggleBaseRequestsPipelineConfig(BaseModel):
        Toggle base requests configuration.
    DataPatternsAccessBehaviorTogglePipelineConfig(BaseModel):
        Toggle behavior configuration.
    DataPatternsAccessBehaviorDistortionNoisePipelineConfig(BaseModel):
        Noise parameters for distortion.
    DataPatternsAccessBehaviorDistortionOffsetsPipelineConfig(BaseModel):
        Distortion offsets configuration.
    DataPatternsAccessBehaviorDistortionPipelineConfig(BaseModel):
        Distortion behavior configuration.
    DataPatternsAccessBehaviorMemoryPipelineConfig(BaseModel):
        Memory behavior configuration.
    DataPatternsAccessBehaviorCyclePipelineConfig(BaseModel):
        Cyclical behavior configuration.
    DataPatternsAccessBehaviorPipelineConfig(BaseModel):
        Aggregated access behavior configuration.
    DataPatternsAccessPipelineConfig(BaseModel):
        Access pattern configuration.
    DataPatternsTemporalBurstinessHoursPipelineConfig(BaseModel):
        Hour range configuration for burstiness.
    DataPatternsTemporalBurstinessPipelineConfig(BaseModel):
        Burstiness configuration.
    DataPatternsTemporalPeriodicPipelineConfig(BaseModel):
        Periodic pattern configuration.
    DataPatternsTemporalPipelineConfig(BaseModel):
        Temporal behavior configuration.
    DataPatternsPipelineConfig(BaseModel):
        Aggregates access and temporal patterns.
    DataGeneralPipelineConfig(BaseModel):
        General configuration for data.
    DataSyntheticPipelineConfig(BaseModel):
        Synthetic data configuration, aggregating mode, and patterns.
    DataPipelineConfig(BaseModel):
        Aggregates general settings and pattern configuration.
"""

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from components.assertions.choice_field_assertor import (
    assert_choice_field,
)
from components.assertions.min_max_assertor import (
    assert_min_less_than_max,
)
from const import (
    DATA_MODES,
    TIME_END_HOUR,
    TIME_START_HOUR,
)


class DataPatternsAccessBehaviorHoursPipelineConfig(BaseModel):
    """Configuration for a range of hours within the day.

    Attributes:
        start (int): Start hour (in [DATA_GENERATION_INITIAL_HOUR,
                     DATA_GENERATION_FINAL_HOUR]).
        end (int): End hour (in [DATA_GENERATION_INITIAL_HOUR,
                   DATA_GENERATION_FINAL_HOUR]).
    """

    start: Annotated[int, Field(ge=TIME_START_HOUR, le=TIME_END_HOUR)]
    end: Annotated[int, Field(ge=TIME_START_HOUR, le=TIME_END_HOUR)]


class DataKeysPipelineConfig(BaseModel):
    """Configuration for key ranges.

    Attributes:
        min (int): Minimum key (> 0).
        max (int): Maximum key (> 0).
    """

    min: Annotated[int, Field(gt=0)]
    max: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def check_min_max_keys(
        self: "DataKeysPipelineConfig",
    ) -> "DataKeysPipelineConfig":
        """Check whether the least key is greater than or equal to the
        greatest key or not.

        Args:
            self (DataKeysPipelineConfig): Current model instance.

        Returns:
            DataKeysPipelineConfig: Validated model instance.
        """
        # Check min/max validity
        assert_min_less_than_max(
            self.min,
            self.max,
            values_context="data.general.keys",
        )

        return self


class DataPatternsAccessZipfAlphaPipelineConfig(BaseModel):
    """Configuration for alpha parameters of Zipf distribution.

    Attributes:
        fixed (float): Fixed alpha value (> 0).
        min (float): Minimum alpha (> 0).
        max (float): Maximum alpha (> 0).
    """

    fixed: Annotated[float, Field(gt=0)]
    min: Annotated[float, Field(gt=0)]
    max: Annotated[float, Field(gt=0)]


class DataPatternsAccessZipfPipelineConfig(BaseModel):
    """Configuration for Zipf distribution.

    Attributes:
        alpha (DataPatternsAccessZipfAlphaPipelineConfig): Alpha configuration.
        steps (int): Number of steps (> 0).
    """

    alpha: DataPatternsAccessZipfAlphaPipelineConfig
    steps: Annotated[int, Field(gt=0)]


class DataPatternsAccessBehaviorRepetitionPipelineConfig(BaseModel):
    """Configuration for repetition behavior in access patterns.

    Attributes:
        interval (int): Interval between repetitions (> 0).
        offset (int): Offset applied to repetitions (> 0).
        hours (DataPatternsAccessBehaviorHoursPipelineConfig):
            Hours during which repetitions occur.
    """

    interval: Annotated[int, Field(gt=0)]
    offset: Annotated[int, Field(gt=0)]
    hours: DataPatternsAccessBehaviorHoursPipelineConfig


class DataPatternsAccessBehaviorToggleOffsetsPipelineConfig(BaseModel):
    """Offset configuration for toggle behavior.

    Attributes:
        forward (int): Forward offset.
        backward (int): Backward offset.
    """

    forward: int
    backward: int


class DataPatternsAccessBehaviorToggleBaseRequestsPipelineConfig(BaseModel):
    """Base requests configuration for toggle behavior.

    Attributes:
        first (int): First base request (> 0).
        second (int): Second base request (> 0).
    """

    first: Annotated[int, Field(gt=0)]
    second: Annotated[int, Field(gt=0)]


class DataPatternsAccessBehaviorTogglePipelineConfig(BaseModel):
    """Configuration for toggle behavior.

    Attributes:
        interval (int): Toggle interval (> 0).
        hours (DataPatternsAccessBehaviorHoursPipelineConfig):
            Hours during which toggle behavior occurs.
        base_requests (DataPatternsAccessBehaviorToggleBaseRequestsPipelineConfig):
            Base request indices.
        offsets (DataPatternsAccessBehaviorToggleOffsetsPipelineConfig):
            Offsets for toggle behavior.
    """

    interval: Annotated[int, Field(gt=0)]
    hours: DataPatternsAccessBehaviorHoursPipelineConfig
    base_requests: DataPatternsAccessBehaviorToggleBaseRequestsPipelineConfig
    offsets: DataPatternsAccessBehaviorToggleOffsetsPipelineConfig


class DataPatternsAccessBehaviorDistortionNoisePipelineConfig(BaseModel):
    """Noise distortion configuration for access behavior.

    Attributes:
        min (int): Minimum noise value.
        max (int): Maximum noise value.
    """

    min: int
    max: int

    @model_validator(mode="after")
    def check_min_max_noises(
        self: "DataPatternsAccessBehaviorDistortionNoisePipelineConfig",
    ) -> "DataPatternsAccessBehaviorDistortionNoisePipelineConfig":
        """Check whether the least noise value is greater than or equal to
        the greatest one or not.

        Args:
            self (DataPatternsAccessBehaviorDistortionNoisePipelineConfig): Current model instance.

        Returns:
            DataPatternsAccessBehaviorDistortionNoisePipelineConfig: Validated model instance.
        """
        assert_min_less_than_max(
            self.min,
            self.max,
            values_context="data.synthetic.patterns.access.behavior.distortion.noise",
        )

        return self


class DataPatternsAccessBehaviorDistortionOffsetsPipelineConfig(BaseModel):
    """Offsets configuration for distortion behavior.

    Attributes:
        history (int): Past history.
        correction (int): Offset for distortion correction.
    """

    history: int
    correction: int


class DataPatternsAccessBehaviorDistortionPipelineConfig(BaseModel):
    """Distortion configuration for access behavior.

    Attributes:
        interval (int): Interval at which distortion is applied (> 0).
        hours (DataPatternsAccessBehaviorHoursPipelineConfig):
            Hours during which distortion occurs.
        offsets (DataPatternsAccessBehaviorDistortionOffsetsPipelineConfig):
            Distortion offsets.
        noise (DataPatternsAccessBehaviorDistortionNoisePipelineConfig):
            Noise parameters for distortion.
    """

    interval: Annotated[int, Field(gt=0)]
    hours: DataPatternsAccessBehaviorHoursPipelineConfig
    offsets: DataPatternsAccessBehaviorDistortionOffsetsPipelineConfig
    noise: DataPatternsAccessBehaviorDistortionNoisePipelineConfig


class DataPatternsAccessBehaviorMemoryPipelineConfig(BaseModel):
    """Memory behavior configuration.

    Attributes:
        interval (int): Interval at which memory is applied (> 0).
        offset (int): Offset applied to memory (> 0).
    """

    interval: Annotated[int, Field(gt=0)]
    offset: Annotated[int, Field(gt=0)]


class DataPatternsAccessBehaviorCyclePipelineConfig(BaseModel):
    """Cyclical behavior configuration.

    Attributes:
        base (int): Base value for cycle (> 0).
        mod (int): Modulus for cycle (> 0).
        divisor (int): Divisor for cycle (> 0).
        hours (DataPatternsAccessBehaviorHoursPipelineConfig):
            Hours during which cyclical behavior occurs.
    """

    base: Annotated[int, Field(gt=0)]
    mod: Annotated[int, Field(gt=0)]
    divisor: Annotated[int, Field(gt=0)]
    hours: DataPatternsAccessBehaviorHoursPipelineConfig


class DataPatternsAccessBehaviorPipelineConfig(BaseModel):
    """Aggregated access behavior configuration.

    Attributes:
        repetition (DataPatternsAccessBehaviorRepetitionPipelineConfig):
            Repetition configuration.
        toggle (DataPatternsAccessBehaviorTogglePipelineConfig):
            Toggle configuration.
        cycle (DataPatternsAccessBehaviorCyclePipelineConfig):
            Cycle configuration.
        distortion (DataPatternsAccessBehaviorDistortionPipelineConfig):
            Distortion configuration.
        memory (DataPatternsAccessBehaviorMemoryPipelineConfig):
            Memory configuration.
    """

    repetition: DataPatternsAccessBehaviorRepetitionPipelineConfig
    toggle: DataPatternsAccessBehaviorTogglePipelineConfig
    cycle: DataPatternsAccessBehaviorCyclePipelineConfig
    distortion: DataPatternsAccessBehaviorDistortionPipelineConfig
    memory: DataPatternsAccessBehaviorMemoryPipelineConfig


class DataPatternsAccessPipelineConfig(BaseModel):
    """Configuration for access patterns.

    Attributes:
        zipf (DataPatternsAccessZipfPipelineConfig): Zipf distribution configuration.
        behavior (DataPatternsAccessBehaviorPipelineConfig): Behavioral access
                                                             configuration.
    """

    zipf: DataPatternsAccessZipfPipelineConfig
    behavior: DataPatternsAccessBehaviorPipelineConfig


class DataPatternsTemporalBurstinessHoursPipelineConfig(BaseModel):
    """Hour range configuration for burstiness.

    Attributes:
        start (int): Start hour of burstiness range (in
                     [DATA_GENERATION_INITIAL_HOUR, DATA_GENERATION_FINAL_HOUR]).
        end (int): End hour of burstiness range (in [DATA_GENERATION_INITIAL_HOUR,
                   DATA_GENERATION_FINAL_HOUR]).
    """

    start: Annotated[int, Field(ge=TIME_START_HOUR, le=TIME_END_HOUR)]
    end: Annotated[int, Field(ge=TIME_START_HOUR, le=TIME_END_HOUR)]


class DataPatternsTemporalBurstinessPipelineConfig(BaseModel):
    """Burstiness configuration for temporal patterns.

    Attributes:
        high (float): High burstiness value (> 0).
        low (float): Low burstiness value (> 0).
        hours (DataPatternsTemporalBurstinessHoursPipelineConfig):
            Hours during which burstiness occurs.
    """

    high: Annotated[float, Field(gt=0.0)]
    low: Annotated[float, Field(gt=0.0)]
    hours: DataPatternsTemporalBurstinessHoursPipelineConfig

    @model_validator(mode="after")
    def check_high_low_bursts(
        self: "DataPatternsTemporalBurstinessPipelineConfig",
    ) -> "DataPatternsTemporalBurstinessPipelineConfig":
        """Check whether the highest burst value is greater than or equal to the
        lowest one or not.

        Args:
            self (DataPatternsTemporalBurstinessPipelineConfig): Current model instance.

        Returns:
            DataPatternsTemporalBurstinessPipelineConfig: Validated model instance.
        """
        # Check min/max validity
        assert_min_less_than_max(
            self.high,
            self.low,
            values_context="data.synthetic.patterns.temporal.burstiness",
        )

        return self


class DataPatternsTemporalPeriodicPipelineConfig(BaseModel):
    """Periodic access pattern configuration.

    Attributes:
        scale (int): Period scale (> 0).
        amplitude (int): Period amplitude (>= 0).
    """

    scale: Annotated[int, Field(gt=0)]
    amplitude: Annotated[int, Field(ge=0)]


class DataPatternsTemporalPipelineConfig(BaseModel):
    """Temporal behavior configuration.

    Attributes:
        burstiness (DataPatternsTemporalBurstinessPipelineConfig):
            Burstiness configuration.
        periodic (DataPatternsTemporalPeriodicPipelineConfig): Periodic pattern
                                                               configuration.
    """

    burstiness: DataPatternsTemporalBurstinessPipelineConfig
    periodic: DataPatternsTemporalPeriodicPipelineConfig


class DataPatternsPipelineConfig(BaseModel):
    """Pattern configuration.

    Attributes:
        access (DataPatternsAccessPipelineConfig): Access pattern configuration.
        temporal (DataPatternsTemporalPipelineConfig): Temporal pattern configuration.
    """

    access: DataPatternsAccessPipelineConfig
    temporal: DataPatternsTemporalPipelineConfig


class DataGeneralPipelineConfig(BaseModel):
    """General configuration for data.

    Attributes:
        requests (int): Number of requests (> 0).
        keys (DataKeysPipelineConfig): Key range configuration.
        mode (str): Data distribution mode.
    """

    requests: Annotated[int, Field(gt=0)]
    keys: DataKeysPipelineConfig
    mode: str

    @model_validator(mode="after")
    def check_data_mode(
        self: "DataSyntheticPipelineConfig",
    ) -> "DataSyntheticPipelineConfig":
        """Check whether data distribution mode is valid or not.

        This function validates the data distribution mode.

        Args:
            self (DataSyntheticPipelineConfig): Current model instance.

        Returns:
            "DataSyntheticConfig": Validated model instance.
        """
        assert_choice_field(
            self.mode,
            DATA_MODES,
            "data.general.mode",
        )

        return self


class DataSyntheticPipelineConfig(BaseModel):
    """Synthetic data configuration.

    Attributes:
        patterns (DataPatternsPipelineConfig): Access and temporal pattern
                                               configuration.
    """

    patterns: DataPatternsPipelineConfig


class DataPipelineConfig(BaseModel):
    """Data configuration.

    Attributes:
        general (DataGeneralPipelineConfig): General data configuration settings.
        synthetic (DataSyntheticPipelineConfig): Synthetic data generation settings
                                                 and patterns.
    """

    general: DataGeneralPipelineConfig
    synthetic: DataSyntheticPipelineConfig
