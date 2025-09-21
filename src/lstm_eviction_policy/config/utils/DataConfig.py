from pydantic import BaseModel, confloat, conint


# Data — Distribution
class RequestsConfig(BaseModel):
    count: conint(gt=0)


class KeysConfig(BaseModel):
    min: conint(ge=0)
    max: conint(gt=0)


class DistributionConfig(BaseModel):
    seed: conint(ge=0)
    mode: str
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
    offset: conint(ge=0)


class ToggleConfig(BaseModel):
    interval: conint(gt=0)


class DistortionConfig(BaseModel):
    interval: conint(gt=0)


class NoiseConfig(BaseModel):
    min: int
    max: int


class MemoryConfig(BaseModel):
    interval: conint(gt=0)
    offset: conint(ge=0)


class CycleConfig(BaseModel):
    base: conint(gt=0)
    mod: conint(gt=0)
    divisor: conint(gt=0)


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
    settings including distribution, pattern,
    sequence, and dataset settings.
    """

    distribution: DistributionConfig
    pattern: PatternConfig
    sequence: SequenceConfig
    dataset: DatasetConfig
