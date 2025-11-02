from pydantic import BaseModel, confloat, conint, model_validator

from components.assertions.choice_field_assertor import (
    assert_choice_field,
)
from components.assertions.min_max_assertor import (
    assert_min_less_than_max,
)
from src.const import (
    DATA_DISTRIBUTION_MODES,
    TIME_END_HOUR,
    TIME_START_HOUR,
)


class DataPatternAccessBehaviorHoursConfig(BaseModel):
    """Configuration for a range of hours within the day.

    Attributes:
        start (int): Start hour (between DATA_GENERATION_INITIAL_HOUR
                     and DATA_GENERATION_FINAL_HOUR).
        end (int): End hour (between DATA_GENERATION_INITIAL_HOUR
                   and DATA_GENERATION_FINAL_HOUR).
    """

    start: conint(
        ge=TIME_START_HOUR,
        le=TIME_END_HOUR,
    )
    end: conint(ge=TIME_START_HOUR, le=TIME_END_HOUR)


class DataKeysConfig(BaseModel):
    """Configuration for key ranges.

    Attributes:
        min (int): Minimum key (> 0).
        max (int): Maximum key (> 0).
    """

    min: conint(gt=0)
    max: conint(gt=0)

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


class DataPatternAccessZipfAlphaConfig(BaseModel):
    """Configuration for alpha parameters of Zipf distribution.

    Attributes:
        fixed (float): Fixed alpha value (> 0).
        min (float): Minimum alpha (> 0).
        max (float): Maximum alpha (> 0).
    """

    fixed: confloat(gt=0)
    min: confloat(gt=0)
    max: confloat(gt=0)


class DataPatternAccessZipfConfig(BaseModel):
    """Configuration for Zipf distribution.

    Attributes:
        alpha (DataPatternAccessZipfAlphaConfig): Alpha configuration.
        steps (int): Number of steps (> 0).
    """

    alpha: DataPatternAccessZipfAlphaConfig
    steps: conint(gt=0)


class DataPatternAccessBehaviorRepetitionConfig(BaseModel):
    """Configuration for repetition behavior in access patterns.

    Attributes:
        interval (int): Interval between repetitions (> 0).
        offset (int): Offset applied to repetitions (> 0).
        hours (DataPatternAccessBehaviorHoursConfig): Hours during which repetitions occur.
    """

    interval: conint(gt=0)
    offset: conint(gt=0)
    hours: DataPatternAccessBehaviorHoursConfig


class DataPatternAccessBehaviorToggleOffsetsConfig(BaseModel):
    """Offset configuration for toggle behavior.

    Attributes:
        forward (int): Forward offset.
        backward (int): Backward offset.
    """

    forward: int
    backward: int


class DataPatternAccessBehaviorToggleBaseRequestsConfig(BaseModel):
    """Base requests configuration for toggle behavior.

    Attributes:
        first (int): First base request (> 0).
        second (int): Second base request (> 0).
    """

    first: conint(gt=0)
    second: conint(gt=0)


class DataPatternAccessBehaviorToggleConfig(BaseModel):
    """Configuration for toggle behavior.

    Attributes:
        interval (int): Toggle interval (> 0).
        hours (DataPatternAccessBehaviorHoursConfig): Hours during which toggle behavior occurs.
        base_requests (DataPatternAccessBehaviorToggleBaseRequestsConfig): Base request indices.
        offsets (DataPatternAccessBehaviorToggleOffsetsConfig): Offsets for toggle behavior.
    """

    interval: conint(gt=0)
    hours: DataPatternAccessBehaviorHoursConfig
    base_requests: DataPatternAccessBehaviorToggleBaseRequestsConfig
    offsets: DataPatternAccessBehaviorToggleOffsetsConfig


class DataPatternAccessBehaviorDistortionNoiseConfig(BaseModel):
    """Noise distortion configuration for access behavior.

    Attributes:
        min (int): Minimum noise value.
        max (int): Maximum noise value.
    """

    min: int
    max: int

    @model_validator(mode="after")
    def check_min_max_noises(
        self: "DataPatternAccessBehaviorDistortionNoiseConfig",
    ) -> "DataPatternAccessBehaviorDistortionNoiseConfig":
        """Check whether the least noise value is greater than or equal to
        the greatest one or not.

        Args:
            self (DataPatternAccessBehaviorDistortionNoiseConfig): Current model instance.

        Returns:
            DataPatternAccessBehaviorDistortionNoiseConfig: Validated model instance.
        """
        assert_min_less_than_max(
            self.min,
            self.max,
            values_context="data.pattern.access.behavior.distortion.noise",
        )

        return self


class DataPatternAccessBehaviorDistortionOffsetsConfig(BaseModel):
    """Offsets configuration for distortion behavior.

    Attributes:
        history (int): Past history.
        correction (int): Offset for distortion correction.
    """

    history: int
    correction: int


class DataPatternAccessBehaviorDistortionConfig(BaseModel):
    """Distortion configuration for access behavior.

    Attributes:
        interval (int): Interval at which distortion is applied (> 0).
        hours (DataPatternAccessBehaviorHoursConfig): Hours during which distortion occurs.
        offsets (DataPatternAccessBehaviorDistortionOffsetsConfig): Distortion offsets.
        noise (DataPatternAccessBehaviorDistortionNoiseConfig): Noise parameters for distortion.
    """

    interval: conint(gt=0)
    hours: DataPatternAccessBehaviorHoursConfig
    offsets: DataPatternAccessBehaviorDistortionOffsetsConfig
    noise: DataPatternAccessBehaviorDistortionNoiseConfig


class DataPatternAccessBehaviorMemoryConfig(BaseModel):
    """Memory behavior configuration.

    Attributes:
        interval (int): Interval at which memory is applied (> 0).
        offset (int): Offset applied to memory (> 0).
    """

    interval: conint(gt=0)
    offset: conint(gt=0)


class DataPatternAccessBehaviorCycleConfig(BaseModel):
    """Cyclical behavior configuration.

    Attributes:
        base (int): Base value for cycle (> 0).
        mod (int): Modulus for cycle (> 0).
        divisor (int): Divisor for cycle (> 0).
        hours (DataPatternAccessBehaviorHoursConfig): Hours during which cyclical behavior occurs.
    """

    base: conint(gt=0)
    mod: conint(gt=0)
    divisor: conint(gt=0)
    hours: DataPatternAccessBehaviorHoursConfig


class DataPatternAccessBehaviorConfig(BaseModel):
    """Aggregated access behavior configuration.

    Attributes:
        repetition (DataPatternAccessBehaviorRepetitionConfig): Repetition configuration.
        toggle (DataPatternAccessBehaviorToggleConfig): Toggle configuration.
        cycle (DataPatternAccessBehaviorCycleConfig): Cycle configuration.
        distortion (DataPatternAccessBehaviorDistortionConfig): Distortion configuration.
        memory (DataPatternAccessBehaviorMemoryConfig): Memory configuration.
    """

    repetition: DataPatternAccessBehaviorRepetitionConfig
    toggle: DataPatternAccessBehaviorToggleConfig
    cycle: DataPatternAccessBehaviorCycleConfig
    distortion: DataPatternAccessBehaviorDistortionConfig
    memory: DataPatternAccessBehaviorMemoryConfig


class DataPatternAccessConfig(BaseModel):
    """Configuration for access patterns.

    Attributes:
        zipf (DataPatternAccessZipfConfig): Zipf distribution configuration.
        behavior (DataPatternAccessBehaviorConfig): Behavioral access configuration.
    """

    zipf: DataPatternAccessZipfConfig
    behavior: DataPatternAccessBehaviorConfig


class DataPatternTemporalBurstinessHoursConfig(BaseModel):
    """Hour range configuration for burstiness.

    Attributes:
        start (int): Start hour of burstiness range (between
            DATA_GENERATION_INITIAL_HOUR and DATA_GENERATION_FINAL_HOUR).
        end (int): End hour of burstiness range (between DATA_GENERATION_INITIAL_HOUR
            and DATA_GENERATION_FINAL_HOUR).
    """

    start: conint(
        ge=TIME_START_HOUR,
        le=TIME_END_HOUR,
    )
    end: conint(ge=TIME_START_HOUR, le=TIME_END_HOUR)


class DataPatternTemporalBurstinessConfig(BaseModel):
    """Burstiness configuration for temporal patterns.

    Attributes:
        high (float): High burstiness value (> 0).
        low (float): Low burstiness value (> 0).
        hours (DataPatternTemporalBurstinessHoursConfig): Hours during which burstiness occurs.
    """

    high: confloat(gt=0)
    low: confloat(gt=0)
    hours: DataPatternTemporalBurstinessHoursConfig

    @model_validator(mode="after")
    def check_high_low_bursts(
        self: "DataPatternTemporalBurstinessConfig",
    ) -> "DataPatternTemporalBurstinessConfig":
        """Check whether the highest burst value is greater than or equal to the
        lowest one or not.

        Args:
            self (DataPatternTemporalBurstinessConfig): Current model instance.

        Returns:
            DataPatternTemporalBurstinessConfig: Validated model instance.
        """
        # Check min/max validity
        assert_min_less_than_max(
            self.high,
            self.low,
            values_context="data.pattern.temporal.burstiness",
        )

        return self


class DataPatternTemporalPeriodicConfig(BaseModel):
    """Periodic access pattern configuration.

    Attributes:
        scale (int): Period scale (> 0).
        amplitude (int): Period amplitude (>= 0).
    """

    scale: conint(gt=0)
    amplitude: conint(ge=0)


class DataPatternTemporalConfig(BaseModel):
    """Temporal behavior configuration.

    Attributes:
        burstiness (DataPatternTemporalBurstinessConfig): Burstiness configuration.
        periodic (DataPatternTemporalPeriodicConfig): Periodic pattern configuration.
    """

    burstiness: DataPatternTemporalBurstinessConfig
    periodic: DataPatternTemporalPeriodicConfig


class DataPatternConfig(BaseModel):
    """Pattern configuration.

    Attributes:
        access (DataPatternAccessConfig): Access pattern configuration.
        temporal (DataPatternTemporalConfig): Temporal pattern configuration.
    """

    access: DataPatternAccessConfig
    temporal: DataPatternTemporalConfig


class DataConfig(BaseModel):
    """Data configuration.

    Attributes:
        seed (int): Random seed for generation (>= 0).
        mode (str): Data distribution mode.
        requests (int): Number of requests (> 0).
        keys (DataKeysConfig): Key range configuration.
        pattern (DataPatternConfig): Pattern configuration.
    """

    seed: conint(ge=0)
    mode: str
    requests: conint(gt=0)
    keys: DataKeysConfig
    pattern: DataPatternConfig

    @model_validator(mode="after")
    def check_data_distribution_mode(
        self: "DataConfig",
    ) -> "DataConfig":
        """Check whether data distribution mode is valid or not.

        This function validates the data distribution mode.

        Args:
            self (DataConfig): Current model instance.

        Returns:
            "DataConfig": Validated model instance.
        """
        assert_choice_field(
            self.mode,
            DATA_DISTRIBUTION_MODES,
            "data.mode",
        )

        return self
