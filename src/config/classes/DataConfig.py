from pydantic import BaseModel, confloat, conint, model_validator

from const import DATA_DISTRIBUTION_MODES, DATA_GENERATION_FINAL_HOUR, DATA_GENERATION_INITIAL_HOUR
from config.classes.validation.choice_field_validator import (
    validate_choice_field,
)
from config.classes.validation.min_max_validator import (
    are_min_max_valid,
)


class HoursConfig(BaseModel):
    start: conint(ge=DATA_GENERATION_INITIAL_HOUR, le=DATA_GENERATION_FINAL_HOUR)  # type: ignore[valid-type]
    end: conint(ge=DATA_GENERATION_INITIAL_HOUR, le=DATA_GENERATION_FINAL_HOUR)  # type: ignore[valid-type]


# Data — Distribution
class RequestsConfig(BaseModel):
    count: conint(gt=0)  # type: ignore[valid-type]


class KeysConfig(BaseModel):
    min: conint(gt=0)  # type: ignore[valid-type]
    max: conint(gt=0)  # type: ignore[valid-type]

    @model_validator(mode="after")
    def check_min_max_keys(
        self: "KeysConfig",
    ) -> "KeysConfig":
        """
        Check whether the least key is greater than
        or equal to the greatest key or not.

        Parameters:
            self (KeysConfig): Current model instance.

        Returns:
            KeysConfig: Validated model instance.
        """
        # Check min/max validity
        are_min_max_valid(
            self.min,
            self.max,
            context="data.general.keys",
        )

        return self


class GeneralDataConfig(BaseModel):
    seed: conint(ge=0)  # type: ignore[valid-type]
    mode: str
    requests: RequestsConfig
    keys: KeysConfig

    @model_validator(mode="after")
    def check_data_distribution_mode(
        self: "GeneralDataConfig",
    ) -> "GeneralDataConfig":
        """
        Check whether data distribution
        mode is valid or not.

        This function validates the data
        distribution mode.

        Parameters:
            self (GeneralDataConfig): Current model instance.

        Returns:
            GeneralDataConfig: Validated model instance.
        """
        return validate_choice_field(
            self,
            self.mode,
            DATA_DISTRIBUTION_MODES,
            "data.general.mode",
        )


# Data — Pattern
class ZipfAlphaConfig(BaseModel):
    fixed: confloat(gt=0)  # type: ignore[valid-type]
    min: confloat(gt=0)  # type: ignore[valid-type]
    max: confloat(gt=0)  # type: ignore[valid-type]


class ZipfConfig(BaseModel):
    alpha: ZipfAlphaConfig
    steps: conint(gt=0)  # type: ignore[valid-type]


class RepetitionConfig(BaseModel):
    interval: conint(gt=0)  # type: ignore[valid-type]
    offset: conint(gt=0)  # type: ignore[valid-type]
    hours: HoursConfig


class ToggleOffsetsConfig(BaseModel):
    forward: int
    backward: int


class ToggleBaseRequestsConfig(BaseModel):
    first: conint(gt=0)  # type: ignore[valid-type]
    second: conint(gt=0)  # type: ignore[valid-type]


class ToggleConfig(BaseModel):
    interval: conint(gt=0)  # type: ignore[valid-type]
    hours: HoursConfig
    base_requests: ToggleBaseRequestsConfig
    offsets: ToggleOffsetsConfig


class DistortionOffsetsConfig(BaseModel):
    history: int
    correction: int


class DistortionConfig(BaseModel):
    interval: conint(gt=0)  # type: ignore[valid-type]
    hours: HoursConfig
    offsets: DistortionOffsetsConfig


class NoiseConfig(BaseModel):
    min: int
    max: int

    @model_validator(mode="after")
    def check_min_max_noises(
        self: "NoiseConfig",
    ) -> "NoiseConfig":
        """
        Check whether the least noise value is
        greater than or equal to the greatest one or not.

        Parameters:
            self (NoiseConfig): Current model instance.

        Returns:
            NoiseConfig: Validated model instance.
        """
        are_min_max_valid(
            self.min,
            self.max,
            context="data.pattern.access.behavior.noise",
        )

        return self


class MemoryConfig(BaseModel):
    interval: conint(gt=0)  # type: ignore[valid-type]
    offset: conint(gt=0)  # type: ignore[valid-type]


class CycleConfig(BaseModel):
    base: conint(gt=0)  # type: ignore[valid-type]
    mod: conint(gt=0)  # type: ignore[valid-type]
    divisor: conint(gt=0)  # type: ignore[valid-type]
    hours: HoursConfig


class BehaviorConfig(BaseModel):
    repetition: RepetitionConfig
    toggle: ToggleConfig
    cycle: CycleConfig
    distortion: DistortionConfig
    noise: NoiseConfig
    memory: MemoryConfig


class AccessConfig(BaseModel):
    zipf: ZipfConfig
    behavior: BehaviorConfig


class BurstinessHoursConfig(BaseModel):
    start: conint(ge=DATA_GENERATION_INITIAL_HOUR, le=DATA_GENERATION_FINAL_HOUR)  # type: ignore[valid-type]
    end: conint(ge=DATA_GENERATION_INITIAL_HOUR, le=DATA_GENERATION_FINAL_HOUR)  # type: ignore[valid-type]


class BurstinessConfig(BaseModel):
    high: confloat(gt=0)  # type: ignore[valid-type]
    low: confloat(gt=0)  # type: ignore[valid-type]
    hours: BurstinessHoursConfig

    @model_validator(mode="after")
    def check_high_low_bursts(
        self: "BurstinessConfig",
    ) -> "BurstinessConfig":
        """
        Check whether the highest burst value is
        greater than or equal to the lowest one or not.

        Parameters:
            self (BurstinessConfig): Current model instance.

        Returns:
            BurstinessConfig: Validated model instance.
        """
        # Check min/max validity
        are_min_max_valid(
            self.high,
            self.low,
            context="data.pattern.temporal.burstiness",
        )

        return self


class PeriodicConfig(BaseModel):
    scale: conint(gt=0)  # type: ignore[valid-type]
    amplitude: conint(ge=0)  # type: ignore[valid-type]


class TemporalConfig(BaseModel):
    burstiness: BurstinessConfig
    periodic: PeriodicConfig


class PatternConfig(BaseModel):
    access: AccessConfig
    temporal: TemporalConfig


# Data — Sequence
class EmbeddingConfig(BaseModel):
    dimension: conint(gt=0)  # type: ignore[valid-type]


class SequenceConfig(BaseModel):
    length: conint(gt=0)  # type: ignore[valid-type]
    embedding: EmbeddingConfig


# Data — Dataset
class SplitConfig(BaseModel):
    training: confloat(ge=0, le=1)  # type: ignore[valid-type]
    validation: confloat(ge=0, le=1)  # type: ignore[valid-type]


class DatasetRawPathsConfig(BaseModel):
    static: str
    dynamic: str


class DatasetProcessedPathsConfig(BaseModel):
    static: str
    dynamic: str


class DatasetPathsConfig(BaseModel):
    raw: DatasetRawPathsConfig
    processed: DatasetProcessedPathsConfig


class DatasetConfig(BaseModel):
    split: SplitConfig
    paths: DatasetPathsConfig


class DataConfig(BaseModel):
    """
    Class representing the data configuration
    settings including general, pattern,
    sequence, and dataset settings.
    """

    general: GeneralDataConfig
    pattern: PatternConfig
    sequence: SequenceConfig
    dataset: DatasetConfig
