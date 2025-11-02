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


class HoursConfig(BaseModel):
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
    )  # type: ignore[valid-type]
    end: conint(ge=TIME_START_HOUR, le=TIME_END_HOUR)  # type: ignore[valid-type]


class KeysConfig(BaseModel):
    """Configuration for key ranges.

    Attributes:
        min (int): Minimum key (> 0).
        max (int): Maximum key (> 0).
    """

    min: conint(gt=0)  # type: ignore[valid-type]
    max: conint(gt=0)  # type: ignore[valid-type]

    @model_validator(mode="after")
    def check_min_max_keys(
        self: "KeysConfig",
    ) -> "KeysConfig":
        """Check whether the least key is greater than or equal to the
        greatest key or not.

        Args:
            self (KeysConfig): Current model instance.

        Returns:
            KeysConfig: Validated model instance.
        """
        # Check min/max validity
        assert_min_less_than_max(
            self.min,
            self.max,
            values_context="data.keys",
        )

        return self


class ZipfAlphaConfig(BaseModel):
    """Configuration for alpha parameters of Zipf distribution.

    Attributes:
        fixed (float): Fixed alpha value (> 0).
        min (float): Minimum alpha (> 0).
        max (float): Maximum alpha (> 0).
    """

    fixed: confloat(gt=0)  # type: ignore[valid-type]
    min: confloat(gt=0)  # type: ignore[valid-type]
    max: confloat(gt=0)  # type: ignore[valid-type]


class ZipfConfig(BaseModel):
    """Configuration for Zipf distribution.

    Attributes:
        alpha (ZipfAlphaConfig): Alpha configuration.
        steps (int): Number of steps (> 0).
    """

    alpha: ZipfAlphaConfig
    steps: conint(gt=0)  # type: ignore[valid-type]


class RepetitionConfig(BaseModel):
    """Configuration for repetition behavior in access patterns.

    Attributes:
        interval (int): Interval between repetitions (> 0).
        offset (int): Offset applied to repetitions (> 0).
        hours (HoursConfig): Hours during which repetitions occur.
    """

    interval: conint(gt=0)  # type: ignore[valid-type]
    offset: conint(gt=0)  # type: ignore[valid-type]
    hours: HoursConfig


class ToggleOffsetsConfig(BaseModel):
    """Offset configuration for toggle behavior.

    Attributes:
        forward (int): Forward offset.
        backward (int): Backward offset.
    """

    forward: int
    backward: int


class ToggleBaseRequestsConfig(BaseModel):
    """Base requests configuration for toggle behavior.

    Attributes:
        first (int): First base request (> 0).
        second (int): Second base request (> 0).
    """

    first: conint(gt=0)  # type: ignore[valid-type]
    second: conint(gt=0)  # type: ignore[valid-type]


class ToggleConfig(BaseModel):
    """Configuration for toggle behavior.

    Attributes:
        interval (int): Toggle interval (> 0).
        hours (HoursConfig): Hours during which toggle behavior occurs.
        base_requests (ToggleBaseRequestsConfig): Base request indices.
        offsets (ToggleOffsetsConfig): Offsets for toggle behavior.
    """

    interval: conint(gt=0)  # type: ignore[valid-type]
    hours: HoursConfig
    base_requests: ToggleBaseRequestsConfig
    offsets: ToggleOffsetsConfig


class NoiseConfig(BaseModel):
    """Noise distortion configuration for access behavior.

    Attributes:
        min (int): Minimum noise value.
        max (int): Maximum noise value.
    """

    min: int
    max: int

    @model_validator(mode="after")
    def check_min_max_noises(
        self: "NoiseConfig",
    ) -> "NoiseConfig":
        """Check whether the least noise value is greater than or equal to
        the greatest one or not.

        Args:
            self (NoiseConfig): Current model instance.

        Returns:
            NoiseConfig: Validated model instance.
        """
        assert_min_less_than_max(
            self.min,
            self.max,
            values_context="data.pattern.access.behavior.distortion.noise",
        )

        return self


class DistortionOffsetsConfig(BaseModel):
    """Offsets configuration for distortion behavior.

    Attributes:
        history (int): Past history.
        correction (int): Offset for distortion correction.
    """

    history: int
    correction: int


class DistortionConfig(BaseModel):
    """Distortion configuration for access behavior.

    Attributes:
        interval (int): Interval at which distortion is applied (> 0).
        hours (HoursConfig): Hours during which distortion occurs.
        offsets (DistortionOffsetsConfig): Distortion offsets.
        noise (NoiseConfig): Noise parameters for distortion.
    """

    interval: conint(gt=0)  # type: ignore[valid-type]
    hours: HoursConfig
    offsets: DistortionOffsetsConfig
    noise: NoiseConfig


class MemoryConfig(BaseModel):
    """Memory behavior configuration.

    Attributes:
        interval (int): Interval at which memory is applied (> 0).
        offset (int): Offset applied to memory (> 0).
    """

    interval: conint(gt=0)  # type: ignore[valid-type]
    offset: conint(gt=0)  # type: ignore[valid-type]


class CycleConfig(BaseModel):
    """Cyclical behavior configuration.

    Attributes:
        base (int): Base value for cycle (> 0).
        mod (int): Modulus for cycle (> 0).
        divisor (int): Divisor for cycle (> 0).
        hours (HoursConfig): Hours during which cyclical behavior occurs.
    """

    base: conint(gt=0)  # type: ignore[valid-type]
    mod: conint(gt=0)  # type: ignore[valid-type]
    divisor: conint(gt=0)  # type: ignore[valid-type]
    hours: HoursConfig


class BehaviorConfig(BaseModel):
    """Aggregated access behavior configuration.

    Attributes:
        repetition (RepetitionConfig): Repetition configuration.
        toggle (ToggleConfig): Toggle configuration.
        cycle (CycleConfig): Cycle configuration.
        distortion (DistortionConfig): Distortion configuration.
        memory (MemoryConfig): Memory configuration.
    """

    repetition: RepetitionConfig
    toggle: ToggleConfig
    cycle: CycleConfig
    distortion: DistortionConfig
    memory: MemoryConfig


class AccessConfig(BaseModel):
    """Configuration for access patterns.

    Attributes:
        zipf (ZipfConfig): Zipf distribution configuration.
        behavior (BehaviorConfig): Behavioral access configuration.
    """

    zipf: ZipfConfig
    behavior: BehaviorConfig


class BurstinessHoursConfig(BaseModel):
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
    )  # type: ignore[valid-type]
    end: conint(ge=TIME_START_HOUR, le=TIME_END_HOUR)  # type: ignore[valid-type]


class BurstinessConfig(BaseModel):
    """Burstiness configuration for temporal patterns.

    Attributes:
        high (float): High burstiness value (> 0).
        low (float): Low burstiness value (> 0).
        hours (BurstinessHoursConfig): Hours during which burstiness occurs.
    """

    high: confloat(gt=0)  # type: ignore[valid-type]
    low: confloat(gt=0)  # type: ignore[valid-type]
    hours: BurstinessHoursConfig

    @model_validator(mode="after")
    def check_high_low_bursts(
        self: "BurstinessConfig",
    ) -> "BurstinessConfig":
        """Check whether the highest burst value is greater than or equal to the
        lowest one or not.

        Args:
            self (BurstinessConfig): Current model instance.

        Returns:
            BurstinessConfig: Validated model instance.
        """
        # Check min/max validity
        assert_min_less_than_max(
            self.high,
            self.low,
            values_context="data.pattern.temporal.burstiness",
        )

        return self


class PeriodicConfig(BaseModel):
    """Periodic access pattern configuration.

    Attributes:
        scale (int): Period scale (> 0).
        amplitude (int): Period amplitude (>= 0).
    """

    scale: conint(gt=0)  # type: ignore[valid-type]
    amplitude: conint(ge=0)  # type: ignore[valid-type]


class TemporalConfig(BaseModel):
    """Temporal behavior configuration.

    Attributes:
        burstiness (BurstinessConfig): Burstiness configuration.
        periodic (PeriodicConfig): Periodic pattern configuration.
    """

    burstiness: BurstinessConfig
    periodic: PeriodicConfig


class PatternConfig(BaseModel):
    """Pattern configuration.

    Attributes:
        access (AccessConfig): Access pattern configuration.
        temporal (TemporalConfig): Temporal pattern configuration.
    """

    access: AccessConfig
    temporal: TemporalConfig


class DataConfig(BaseModel):
    """Data configuration.

    Attributes:
        seed (int): Random seed for generation (>= 0).
        mode (str): Data distribution mode.
        requests (int): Number of requests (> 0).
        keys (KeysConfig): Key range configuration.
        pattern (PatternConfig): Pattern configuration.
    """

    seed: conint(ge=0)  # type: ignore[valid-type]
    mode: str
    requests: conint(gt=0)  # type: ignore[valid-type]
    keys: KeysConfig
    pattern: PatternConfig

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
