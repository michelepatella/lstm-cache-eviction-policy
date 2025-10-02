from pydantic import BaseModel, confloat, conint


# Inference — Confidence Intervals
class ConfidenceIntervalsConfig(BaseModel):
    level: confloat(ge=0, le=1)  # type: ignore[valid-type]


# Inference — MC Dropout
class MCDropoutSamplesConfig(BaseModel):
    count: conint(gt=0)  # type: ignore[valid-type]


class MCDropoutConfig(BaseModel):
    samples: MCDropoutSamplesConfig


class InferenceConfig(BaseModel):
    """
    Class representing the inference configuration
    settings including confidence intervals and
    MC dropout settings.
    """

    confidence_intervals: ConfidenceIntervalsConfig
    mc_dropout: MCDropoutConfig
