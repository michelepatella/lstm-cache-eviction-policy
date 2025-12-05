"""data_config.py

Configuration section for data parameters.

This module structures all parameters necessary for generating synthetic
request data, covering core aspects like key ranges, total request count,
distribution mode, and complex temporal and access patterns.

It defines parameters for Zipf distribution, burstiness, and various access
behaviors (repetition, toggle, cycle, distortion, and memory effects).

Classes:
    DataPatternsAccessBehaviorHoursConfig(BaseModel):
        Hour range configuration.
    DataKeysConfig(BaseModel):
        Configuration for key ID ranges.
    DataPatternsAccessZipfAlphaConfig(BaseModel):
        Alpha parameters for Zipf distribution.
    DataPatternsAccessZipfConfig(BaseModel):
        Configuration for Zipf distribution.
    DataPatternsAccessBehaviorRepetitionConfig(BaseModel):
        Repetition behavior configuration.
    DataPatternsAccessBehaviorToggleOffsetsConfig(BaseModel):
        Toggle offsets configuration.
    DataPatternsAccessBehaviorToggleBaseRequestsConfig(BaseModel):
        Toggle base requests configuration.
    DataPatternsAccessBehaviorToggleConfig(BaseModel):
        Toggle behavior configuration.
    DataPatternsAccessBehaviorDistortionNoiseConfig(BaseModel):
        Noise parameters for distortion.
    DataPatternsAccessBehaviorDistortionOffsetsConfig(BaseModel):
        Distortion offsets configuration.
    DataPatternsAccessBehaviorDistortionConfig(BaseModel):
        Distortion behavior configuration.
    DataPatternsAccessBehaviorMemoryConfig(BaseModel):
        Memory behavior configuration.
    DataPatternsAccessBehaviorCycleConfig(BaseModel):
        Cyclical behavior configuration.
    DataPatternsAccessBehaviorConfig(BaseModel):
        Aggregated access behavior configuration.
    DataPatternsAccessConfig(BaseModel):
        Access pattern configuration.
    DataPatternsTemporalBurstinessHoursConfig(BaseModel):
        Hour range configuration for burstiness.
    DataPatternsTemporalBurstinessConfig(BaseModel):
        Burstiness configuration.
    DataPatternsTemporalPeriodicConfig(BaseModel):
        Periodic pattern configuration.
    DataPatternsTemporalConfig(BaseModel):
        Temporal behavior configuration.
    DataPatternsConfig(BaseModel):
        Aggregates access and temporal patterns.
    DataGeneralConfig(BaseModel):
        General configuration for data.
    DataSyntheticConfig(BaseModel):
        Synthetic data configuration, aggregating mode, patterns, and seed.
    DataConfig(BaseModel):
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


class DataPatternsAccessBehaviorHoursConfig(BaseModel):
    """Configuration for a range of hours within the day.

    Attributes:
        start (int): Start hour (between DATA_GENERATION_INITIAL_HOUR
                     and DATA_GENERATION_FINAL_HOUR).
        end (int): End hour (between DATA_GENERATION_INITIAL_HOUR
                   and DATA_GENERATION_FINAL_HOUR).
    """

    start: Annotated[int, Field(ge=TIME_START_HOUR, le=TIME_END_HOUR)]
    end: Annotated[int, Field(ge=TIME_START_HOUR, le=TIME_END_HOUR)]


class DataKeysConfig(BaseModel):
    """Configuration for key ranges.

    Attributes:
        min (int): Minimum key (> 0).
        max (int): Maximum key (> 0).
    """

    min: Annotated[int, Field(gt=0)]
    max: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def check_min_max_keys(
        self: "DataKeysConfig",
    ) -> "DataKeysConfig":
        """Check whether the least key is greater than or equal to the
        greatest key or not.

        Args:
            self (DataKeysConfig): Current model instance.

        Returns:
            DataKeysConfig: Validated model instance.
        """
        # Check min/max validity
        assert_min_less_than_max(
            self.min,
            self.max,
            values_context="data.keys",
        )

        return self


class DataPatternsAccessZipfAlphaConfig(BaseModel):
    """Configuration for alpha parameters of Zipf distribution.

    Attributes:
        fixed (float): Fixed alpha value (> 0).
        min (float): Minimum alpha (> 0).
        max (float): Maximum alpha (> 0).
    """

    fixed: Annotated[float, Field(gt=0)]
    min: Annotated[float, Field(gt=0)]
    max: Annotated[float, Field(gt=0)]


class DataPatternsAccessZipfConfig(BaseModel):
    """Configuration for Zipf distribution.

    Attributes:
        alpha (DataPatternsAccessZipfAlphaConfig): Alpha configuration.
        steps (int): Number of steps (> 0).
    """

    alpha: DataPatternsAccessZipfAlphaConfig
    steps: Annotated[int, Field(gt=0)]


class DataPatternsAccessBehaviorRepetitionConfig(BaseModel):
    """Configuration for repetition behavior in access patterns.

    Attributes:
        interval (int): Interval between repetitions (> 0).
        offset (int): Offset applied to repetitions (> 0).
        hours (DataPatternsAccessBehaviorHoursConfig): Hours during which repetitions occur.
    """

    interval: Annotated[int, Field(gt=0)]
    offset: Annotated[int, Field(gt=0)]
    hours: DataPatternsAccessBehaviorHoursConfig


class DataPatternsAccessBehaviorToggleOffsetsConfig(BaseModel):
    """Offset configuration for toggle behavior.

    Attributes:
        forward (int): Forward offset.
        backward (int): Backward offset.
    """

    forward: int
    backward: int


class DataPatternsAccessBehaviorToggleBaseRequestsConfig(BaseModel):
    """Base requests configuration for toggle behavior.

    Attributes:
        first (int): First base request (> 0).
        second (int): Second base request (> 0).
    """

    first: Annotated[int, Field(gt=0)]
    second: Annotated[int, Field(gt=0)]


class DataPatternsAccessBehaviorToggleConfig(BaseModel):
    """Configuration for toggle behavior.

    Attributes:
        interval (int): Toggle interval (> 0).
        hours (DataPatternsAccessBehaviorHoursConfig): Hours during which toggle behavior occurs.
        base_requests (DataPatternsAccessBehaviorToggleBaseRequestsConfig): Base request indices.
        offsets (DataPatternsAccessBehaviorToggleOffsetsConfig): Offsets for toggle behavior.
    """

    interval: Annotated[int, Field(gt=0)]
    hours: DataPatternsAccessBehaviorHoursConfig
    base_requests: DataPatternsAccessBehaviorToggleBaseRequestsConfig
    offsets: DataPatternsAccessBehaviorToggleOffsetsConfig


class DataPatternsAccessBehaviorDistortionNoiseConfig(BaseModel):
    """Noise distortion configuration for access behavior.

    Attributes:
        min (int): Minimum noise value.
        max (int): Maximum noise value.
    """

    min: int
    max: int

    @model_validator(mode="after")
    def check_min_max_noises(
        self: "DataPatternsAccessBehaviorDistortionNoiseConfig",
    ) -> "DataPatternsAccessBehaviorDistortionNoiseConfig":
        """Check whether the least noise value is greater than or equal to
        the greatest one or not.

        Args:
            self (DataPatternsAccessBehaviorDistortionNoiseConfig): Current model instance.

        Returns:
            DataPatternsAccessBehaviorDistortionNoiseConfig: Validated model instance.
        """
        assert_min_less_than_max(
            self.min,
            self.max,
            values_context="data.pattern.access.behavior.distortion.noise",
        )

        return self


class DataPatternsAccessBehaviorDistortionOffsetsConfig(BaseModel):
    """Offsets configuration for distortion behavior.

    Attributes:
        history (int): Past history.
        correction (int): Offset for distortion correction.
    """

    history: int
    correction: int


class DataPatternsAccessBehaviorDistortionConfig(BaseModel):
    """Distortion configuration for access behavior.

    Attributes:
        interval (int): Interval at which distortion is applied (> 0).
        hours (DataPatternsAccessBehaviorHoursConfig): Hours during which distortion occurs.
        offsets (DataPatternsAccessBehaviorDistortionOffsetsConfig): Distortion offsets.
        noise (DataPatternsAccessBehaviorDistortionNoiseConfig): Noise parameters for distortion.
    """

    interval: Annotated[int, Field(gt=0)]
    hours: DataPatternsAccessBehaviorHoursConfig
    offsets: DataPatternsAccessBehaviorDistortionOffsetsConfig
    noise: DataPatternsAccessBehaviorDistortionNoiseConfig


class DataPatternsAccessBehaviorMemoryConfig(BaseModel):
    """Memory behavior configuration.

    Attributes:
        interval (int): Interval at which memory is applied (> 0).
        offset (int): Offset applied to memory (> 0).
    """

    interval: Annotated[int, Field(gt=0)]
    offset: Annotated[int, Field(gt=0)]


class DataPatternsAccessBehaviorCycleConfig(BaseModel):
    """Cyclical behavior configuration.

    Attributes:
        base (int): Base value for cycle (> 0).
        mod (int): Modulus for cycle (> 0).
        divisor (int): Divisor for cycle (> 0).
        hours (DataPatternsAccessBehaviorHoursConfig): Hours during which cyclical behavior occurs.
    """

    base: Annotated[int, Field(gt=0)]
    mod: Annotated[int, Field(gt=0)]
    divisor: Annotated[int, Field(gt=0)]
    hours: DataPatternsAccessBehaviorHoursConfig


class DataPatternsAccessBehaviorConfig(BaseModel):
    """Aggregated access behavior configuration.

    Attributes:
        repetition (DataPatternsAccessBehaviorRepetitionConfig): Repetition configuration.
        toggle (DataPatternsAccessBehaviorToggleConfig): Toggle configuration.
        cycle (DataPatternsAccessBehaviorCycleConfig): Cycle configuration.
        distortion (DataPatternsAccessBehaviorDistortionConfig): Distortion configuration.
        memory (DataPatternsAccessBehaviorMemoryConfig): Memory configuration.
    """

    repetition: DataPatternsAccessBehaviorRepetitionConfig
    toggle: DataPatternsAccessBehaviorToggleConfig
    cycle: DataPatternsAccessBehaviorCycleConfig
    distortion: DataPatternsAccessBehaviorDistortionConfig
    memory: DataPatternsAccessBehaviorMemoryConfig


class DataPatternsAccessConfig(BaseModel):
    """Configuration for access patterns.

    Attributes:
        zipf (DataPatternsAccessZipfConfig): Zipf distribution configuration.
        behavior (DataPatternsAccessBehaviorConfig): Behavioral access configuration.
    """

    zipf: DataPatternsAccessZipfConfig
    behavior: DataPatternsAccessBehaviorConfig


class DataPatternsTemporalBurstinessHoursConfig(BaseModel):
    """Hour range configuration for burstiness.

    Attributes:
        start (int): Start hour of burstiness range (between
            DATA_GENERATION_INITIAL_HOUR and DATA_GENERATION_FINAL_HOUR).
        end (int): End hour of burstiness range (between DATA_GENERATION_INITIAL_HOUR
            and DATA_GENERATION_FINAL_HOUR).
    """

    start: Annotated[int, Field(ge=TIME_START_HOUR, le=TIME_END_HOUR)]
    end: Annotated[int, Field(ge=TIME_START_HOUR, le=TIME_END_HOUR)]


class DataPatternsTemporalBurstinessConfig(BaseModel):
    """Burstiness configuration for temporal patterns.

    Attributes:
        high (float): High burstiness value (> 0).
        low (float): Low burstiness value (> 0).
        hours (DataPatternsTemporalBurstinessHoursConfig): Hours during which burstiness occurs.
    """

    high: Annotated[float, Field(gt=0.0)]
    low: Annotated[float, Field(gt=0.0)]
    hours: DataPatternsTemporalBurstinessHoursConfig

    @model_validator(mode="after")
    def check_high_low_bursts(
        self: "DataPatternsTemporalBurstinessConfig",
    ) -> "DataPatternsTemporalBurstinessConfig":
        """Check whether the highest burst value is greater than or equal to the
        lowest one or not.

        Args:
            self (DataPatternsTemporalBurstinessConfig): Current model instance.

        Returns:
            DataPatternsTemporalBurstinessConfig: Validated model instance.
        """
        # Check min/max validity
        assert_min_less_than_max(
            self.high,
            self.low,
            values_context="data.pattern.temporal.burstiness",
        )

        return self


class DataPatternsTemporalPeriodicConfig(BaseModel):
    """Periodic access pattern configuration.

    Attributes:
        scale (int): Period scale (> 0).
        amplitude (int): Period amplitude (>= 0).
    """

    scale: Annotated[int, Field(gt=0)]
    amplitude: Annotated[int, Field(ge=0)]


class DataPatternsTemporalConfig(BaseModel):
    """Temporal behavior configuration.

    Attributes:
        burstiness (DataPatternsTemporalBurstinessConfig): Burstiness configuration.
        periodic (DataPatternsTemporalPeriodicConfig): Periodic pattern configuration.
    """

    burstiness: DataPatternsTemporalBurstinessConfig
    periodic: DataPatternsTemporalPeriodicConfig


class DataPatternsConfig(BaseModel):
    """Pattern configuration.

    Attributes:
        access (DataPatternsAccessConfig): Access pattern configuration.
        temporal (DataPatternsTemporalConfig): Temporal pattern configuration.
    """

    access: DataPatternsAccessConfig
    temporal: DataPatternsTemporalConfig


class DataGeneralConfig(BaseModel):
    """General configuration for data.

    Attributes:
        requests (int): Number of requests (> 0).
        keys (DataKeysConfig): Key range configuration.
        mode (str): Data distribution mode.
    """

    requests: Annotated[int, Field(gt=0)]
    keys: DataKeysConfig
    mode: str

    @model_validator(mode="after")
    def check_data_mode(
        self: "DataSyntheticConfig",
    ) -> "DataSyntheticConfig":
        """Check whether data distribution mode is valid or not.

        This function validates the data distribution mode.

        Args:
            self (DataSyntheticConfig): Current model instance.

        Returns:
            "DataSyntheticConfig": Validated model instance.
        """
        assert_choice_field(
            self.mode,
            DATA_MODES,
            "data.general.mode",
        )

        return self


class DataSyntheticConfig(BaseModel):
    """Synthetic data configuration.

    Attributes:
        patterns (DataPatternsConfig): Access and temporal pattern configuration.
        seed (int): Random seed for generation (>= 0).
    """

    patterns: DataPatternsConfig
    seed: Annotated[int, Field(ge=0)]


class DataConfig(BaseModel):
    """Data configuration.

    Attributes:
        general (DataGeneralConfig): General data configuration settings.
        synthetic (DataSyntheticConfig): Synthetic data generation settings and patterns.
    """

    general: DataGeneralConfig
    synthetic: DataSyntheticConfig
