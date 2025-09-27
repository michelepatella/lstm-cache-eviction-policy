from pydantic import BaseModel, conint


class EvaluationConfig(BaseModel):
    """
    Class representing the evaluation configuration
    settings.
    """

    top_k: conint(gt=0)
