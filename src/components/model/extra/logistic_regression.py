from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


@dataclass
class LogisticRegressionWrapper:
    """This class wraps the logistic regression model.

    Attributes:
        model: The logistic regression model wrapped.
        encoder: One-hot encoder used for keys.
        seq_len: Sequence length of embedded keys.
    """

    model: LogisticRegression
    encoder: OneHotEncoder
    seq_len: int
