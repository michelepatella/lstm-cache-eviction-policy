import torch
from torch import nn

from components.const import (
    DATASET_TARGET_COLUMN_IDX,
    MC_DROPOUT_DISABLED,
    MODEL_PARAM_NAMES,
)
from components.device.mover import move_to_device
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from pipeline.config.pydantic.config import Config
from pipeline.config.pydantic.sections.model_config import ModelParamsConfig


class LSTM(torch.nn.Module):
    """Long Short Term Memory (LSTM) model.

    This class implements an LSTM model to predict the probability
    of accessing a key next.

    Attributes:
        mc_dropout (bool): Flag indicating if MC dropout is enabled or not.
        num_keys (int): Number of unique keys to embed.
        input_size (int): Size of model input.
        num_features (int): Number of features in input data.
        embedding_dim (int): Dimensionality of key embeddings.
        embedding (nn.Embedding): Embedding layer for keys.
        lstm (nn.LSTM): PyTorch LSTM module.
        mc_dropout_layer (nn.Dropout): Dropout layer for MC dropout.
        fc (nn.Linear): Fully connected layer.
    """

    def _set_params(
        self: "LSTM",
        params: ModelParamsConfig | dict[str, int | float | bool],
        config: Config | None,
        param_names: list[str] = MODEL_PARAM_NAMES,
    ) -> None:
        """Set model parameters.

        This function sets model parameters by using values provided directly,
        or falling back to configuration values if not provided.

        Args:
            self ("LSTM"): Current model instance.
            params (ModelParamsConfig |
            dict[str, int | float | bool]): Model parameters to be set.
            config (Config | None): Configuration object.
            param_names (list[str]): List of parameter names to set.

        Raises:
            RuntimeError: If setting model parameters fails:
                * Accessing parameter values in the configuration fails due to
                  missing attributes (AttributeError).
                * Setting a model attribute fails due to invalid type
                  (TypeError).
                * A provided or configuration value is invalid (ValueError).
        """
        try:
            # Prepare configuration model params
            # if the configuration object is passed
            model_params = None
            if config is not None:
                model_params = config.model.params

            # For each required parameter
            for param in param_names:
                # Check whether the required parameter
                # has been directly passed to the model
                if param in params and params[param] is not None:
                    # Set the specified parameter value
                    setattr(self, param, params[param])
                else:
                    # Otherwise, retrieve parameter value from
                    # configuration
                    config_value = getattr(model_params, param)

                    # Set the value specified by configuration
                    setattr(self, param, config_value)
        except (AttributeError, TypeError, ValueError) as e:
            msg = "Model parameters setting failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "model_type": type(self).__name__,
                    "param_names_attempted": param_names,
                    "params_passed": params,
                    "context": "LSTM model",
                },
            )
            raise RuntimeError(msg) from e

    def _set_fields(
        self: "LSTM",
        params: ModelParamsConfig | dict[str, int | float | bool],
        min_key: int,
        max_key: int,
        embedding_dim: int,
        num_features: int,
        config: Config,
    ) -> None:
        """Set model fields.

        This function sets model fields.

        Args:
            self ("LSTM"): Current class instance.
            params (ModelParamsConfig |
            dict[str, int | float | bool]): Model parameters.
            min_key (int): Maximum key.
            max_key (int): Minimum key.
            embedding_dim (int): Embedding dimension for keys.
            num_features (int): Number of features for the model.
            config (Config): Configuration object.

        Returns:
            None
        """
        # Set model params
        self._set_params(params, config)

        # Disable MC Dropout by default
        self.mc_dropout = MC_DROPOUT_DISABLED

        # Set number of keys
        self.num_keys = max_key - min_key + 1

        # Set model input size
        self.input_size = num_features + embedding_dim

        # Set number of features
        self.num_features = num_features

        # Set embedding with specified dimension
        self.embedding_dim = embedding_dim

        debug(
            "Model fields setting executed",
            extra={
                "model_type": type(self).__name__,
                "keys_num": self.num_keys,
                "input_size": self.input_size,
                "features_num": self.num_features,
                "embedding_dim": self.embedding_dim,
                "mc_dropout_enabled": self.mc_dropout,
                "context": "LSTM model",
            },
        )

    def _set_layers(self: "LSTM") -> None:
        """Instantiate all the layers of the model.

        This function creates and assigns the following layers to the model:
            - Embedding layer
            - Dropout layer
            - Fully connected (linear) layer

        Args:
            self ("LSTM"): Current class instance.

        Returns:
            None

        Raises:
        RuntimeError: If layer instantiation fails:
            * Embedding layer creation fails due to invalid number of keys or
              embedding dimension (TypeError, ValueError, AttributeError).
            * Dropout layer creation fails due to invalid dropout probability
              (TypeError, ValueError, AttributeError).
            * Fully connected layer creation fails due to invalid hidden size
              or number of keys (TypeError, ValueError, AttributeError).
        """
        try:
            # Instantiate embedding layer
            self.embedding = nn.Embedding(self.num_keys, self.embedding_dim)

            # Instantiate dropout layer
            self.mc_dropout_layer = nn.Dropout(p=self.dropout)

            # Instantiate fully connected layer
            self.fc = nn.Linear(self.hidden_size, self.num_keys)

            debug(
                "Model layers setting executed",
                extra={
                    "model_type": type(self).__name__,
                    "embedding_shape": (self.num_keys, self.embedding_dim),
                    "dropout_prob": self.dropout,
                    "fc_in_features": self.hidden_size,
                    "fc_out_features": self.num_keys,
                    "context": "LSTM model",
                },
            )
        except (TypeError, ValueError, AttributeError) as e:
            msg = "Model layers setting failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "model_type": type(self).__name__,
                    "keys_num": getattr(self, "num_keys", None),
                    "embedding_dim": getattr(self, "embedding_dim", None),
                    "hidden_size": getattr(self, "hidden_size", None),
                    "context": "LSTM model",
                },
            )
            raise RuntimeError(msg) from e

    def __init__(
        self: "LSTM",
        params: ModelParamsConfig | dict[str, int | float | bool],
        min_key: int,
        max_key: int,
        embedding_dim: int,
        num_features: int,
        config: Config | None,
    ) -> None:
        """Initialize the model.

        This function initializes the model, specifically:
        sets all fields, instantiates the class, as well as all its layers.

        Args:
            self ("LSTM"): Current class instance.
            params (ModelParamsConfig |
            dict[str, int | float | bool]): Model parameters.
            min_key (int): Maximum key.
            max_key (int): Minimum key.
            embedding_dim (int): Embedding dimension for keys.
            num_features (int): Number of features for the model.
            config (Config | None): Configuration object.

        Returns:
            None
        """
        super().__init__()

        # Set model fields
        self._set_fields(
            params,
            min_key,
            max_key,
            embedding_dim,
            num_features,
            config,
        )

        # Instantiate the LSTM model
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=self.bias,
            batch_first=self.batch_first,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            proj_size=self.proj_size,
        )

        # Set all model layers
        self._set_layers()

        debug(
            "LSTM model initialization executed",
            extra={
                "model_type": type(self).__name__,
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "layers_num": self.num_layers,
                "bias": self.bias,
                "batch_first": self.batch_first,
                "dropout": self.dropout,
                "bidirectional": self.bidirectional,
                "proj_size": self.proj_size,
                "keys_num": getattr(self, "num_keys", None),
                "embedding_dim": getattr(self, "embedding_dim", None),
                "features_num": getattr(self, "num_features", None),
                "mc_dropout_enabled": getattr(self, "mc_dropout", None),
                "context": "LSTM model",
            },
        )

    def _build_model_input(
        self: "LSTM",
        x_features: torch.Tensor,
        x_keys: torch.Tensor,
    ) -> torch.Tensor:
        """Build and return the model input.

        This function builds and returns the model input by concatenating
        features with embedded keys.

        Args:
            self ("LSTM"): Current class instance.
            x_features (torch.Tensor): Feature tensor for the current batch.
            x_keys (torch.Tensor): Keys tensor for the current batch.

        Returns:
            torch.Tensor: Concatenated input tensor ready for model.
        """
        try:
            # Get the device of embedding layer
            device = self.embedding.weight.device

            # Ensure keys are on the same device
            # as the embedding layer
            x_features = move_to_device(x_features, device)
            x_keys = move_to_device(x_keys, device)

            # Perform embedding
            x_keys = x_keys.long()
            embedded_keys = self.embedding(x_keys)

            # Concatenate features with embedded keys
            x = torch.cat(
                (x_features, embedded_keys),
                dim=DATASET_TARGET_COLUMN_IDX,
            )

            return x
        except (AttributeError, TypeError, RuntimeError) as e:
            msg = "Model input building failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "x_features_shape": getattr(x_features, "shape", None),
                    "x_keys_shape": getattr(x_keys, "shape", None),
                    "embedding_device": getattr(
                        self.embedding.weight,
                        "device",
                        None,
                    ),
                    "context": "LSTM model",
                },
            )
            raise RuntimeError(msg) from e

    def forward(
        self: "LSTM",
        x_features: torch.Tensor,
        x_keys: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a forward pass through the model.

        This function builds the model input by concatenating features
        with embedded keys, passes it through the model, optionally
        applies MC dropout, and finally computes logits using a linear layer.

        Args:
            x_features (torch.Tensor): Input features tensor for
                                       the current batch.
            x_keys (torch.Tensor): Input keys tensor for the current batch.

        Returns:
            torch.Tensor: Logits computed by the model.
        """
        try:
            # Build model input for current batch
            x = self._build_model_input(x_features, x_keys)

            # Pass the input through the model
            output, _ = self.lstm(x)

            # Apply MC dropout if enabled
            if self.mc_dropout:
                output = self.mc_dropout_layer(output)

            # Compute logits from last step
            logits = self.fc(output[:, -1, :])

            return logits
        except (IndexError, RuntimeError) as e:
            msg = "Model forward pass failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "x_features_shape": getattr(x_features, "shape", None),
                    "x_keys_shape": getattr(x_keys, "shape", None),
                    "mc_dropout_enabled": getattr(self, "mc_dropout", None),
                    "lstm_input_size": getattr(self.lstm, "input_size", None),
                    "fc_output_size": getattr(self.fc, "out_features", None),
                    "context": "LSTM model",
                },
            )
            raise RuntimeError(msg) from e
