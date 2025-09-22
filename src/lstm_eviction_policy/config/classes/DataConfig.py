from typing import Literal

from pydantic import BaseModel, confloat, conint, model_validator

from lstm_eviction_policy.config.utils.min_max_validator import validate_min_max


class HoursConfig(BaseModel):
    start: conint(ge=0, le=23)
    end: conint(ge=0, le=23)


# Data — Distribution
class RequestsConfig(BaseModel):
    count: conint(gt=0)


class KeysConfig(BaseModel):
    min: conint(gt=0)
    max: conint(gt=0)

    @model_validator(mode="after")
    def check_min_max_keys(self: "KeysConfig") -> "KeysConfig":
        """
        Check whether the least key is greater than
        or equal to the greatest key or not.

        Parameters:
            self (KeysConfig): Current model instance.

        Returns:
            KeysConfig: Validated model instance.
        """
        return validate_min_max(self, "min", "max", context="data.general.keys.")


class GeneralDataConfig(BaseModel):
    seed: conint(ge=0)
    mode: Literal["static", "dynamic"]
    requests: RequestsConfig
    keys: KeysConfig


# Data — Pattern
class ZipfAlphaConfig(BaseModel):
    fixed: confloat(gt=0)
    min: confloat(gt=0)
    max: confloat(gt=0)


class ZipfConfig(BaseModel):
    alpha: ZipfAlphaConfig
    steps: conint(gt=0)


class RepetitionConfig(BaseModel):
    interval: conint(gt=0)
    offset: conint(gt=0)
    hours: HoursConfig


class ToggleOffsetsConfig(BaseModel):
    forward: int
    backward: int


class ToggleBaseRequestsConfig(BaseModel):
    first: conint(gt=0)
    second: conint(gt=0)


class ToggleConfig(BaseModel):
    interval: conint(gt=0)
    hours: HoursConfig
    base_requests: ToggleBaseRequestsConfig
    offsets: ToggleOffsetsConfig


class DistortionOffsetsConfig(BaseModel):
    history: int
    correction: int


class DistortionConfig(BaseModel):
    interval: conint(gt=0)
    hours: HoursConfig
    offsets: DistortionOffsetsConfig


class NoiseConfig(BaseModel):
    min: int
    max: int

    @model_validator(mode="after")
    def check_min_max_noises(self: "NoiseConfig") -> "NoiseConfig":
        """
        Check whether the least noise value is
        greater than or equal to the greatest one or not.

        Parameters:
            self (NoiseConfig): Current model instance.

        Returns:
            NoiseConfig: Validated model instance.
        """
        return validate_min_max(
            self, "min", "max", context="data.pattern.access.behavior.noise."
        )


class MemoryConfig(BaseModel):
    interval: conint(gt=0)
    offset: conint(gt=0)


class CycleConfig(BaseModel):
    base: conint(gt=0)
    mod: conint(gt=0)
    divisor: conint(gt=0)
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
    start: conint(ge=0, le=23)
    end: conint(ge=0, le=23)


class BurstinessConfig(BaseModel):
    high: confloat(gt=0)
    low: confloat(gt=0)
    hours: BurstinessHoursConfig

    @model_validator(mode="after")
    def check_high_low_bursts(self: "BurstinessConfig") -> "BurstinessConfig":
        """
        Check whether the highest burst value is
        greater than or equal to the lowest one or not.

        Parameters:
            self (BurstinessConfig): Current model instance.

        Returns:
            BurstinessConfig: Validated model instance.
        """
        return validate_min_max(
            self, "high", "low", context="data.pattern.temporal.burstiness."
        )


class PeriodicConfig(BaseModel):
    scale: conint(gt=0)
    amplitude: conint(ge=0)


class TemporalConfig(BaseModel):
    burstiness: BurstinessConfig
    periodic: PeriodicConfig


class PatternConfig(BaseModel):
    access: AccessConfig
    temporal: TemporalConfig


# Data — Sequence
class EmbeddingConfig(BaseModel):
    dimension: conint(gt=0)


class SequenceConfig(BaseModel):
    length: conint(gt=0)
    embedding: EmbeddingConfig


# Data — Dataset
class SplitConfig(BaseModel):
    train: confloat(ge=0, le=1)
    validation: confloat(ge=0, le=1)


class DatasetPathsConfig(BaseModel):
    static: str
    dynamic: str


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
