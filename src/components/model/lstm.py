from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from components.const import MC_DROPOUT_DISABLED, MODEL_PARAM_NAMES
from components.device.mover import move_to_device
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from pipeline.config.pydantic.config import Config
from pipeline.config.pydantic.sections.model_config import ModelParamsConfig


class LSTM(torch.nn.Module):
    """
    Long Short Term Memory (LSTM) model.

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
        params: Union[ModelParamsConfig, Dict[str, Union[int, float, bool]]],
        config: Optional[Config],
        param_names: List[str] = MODEL_PARAM_NAMES,
    ) -> None:
        """
        Set model parameters.

        This function sets model parameters by using values provided directly,
        or falling back to configuration values if not provided.

        Args:
            self ("LSTM"): Current model instance.
            params (Union[ModelParamsConfig, Dict[str, Union[int, float, bool]]]):
                Model parameters to be set.
            config (Optional[Config]): Configuration object.
            param_names (List[str]): List of parameter names to set.

        Raises:
            RuntimeError: If setting model parameters fails:
                * Accessing parameter values in the configuration fails due to
                  missing attributes (AttributeError).
                * Setting a model attribute fails due to invalid type (TypeError).
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
                    debug(f"Model parameter '{param}' passed: {params[param]}")

                    # Set the specified parameter value
                    setattr(self, param, params[param])
                else:
                    # Otherwise, retrieve parameter value from
                    # configuration
                    config_value = getattr(model_params, param)
                    debug(
                        f"Model parameter '{param}' not passed,"
                        f" using configuration value: {config_value}"
                    )

                    # Set the value specified by configuration
                    setattr(self, param, config_value)

            debug("Model parameters set")
        except (AttributeError, TypeError, ValueError) as e:
            msg = "Failed to set model parameters"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def _set_fields(
        self: "LSTM",
        params: Union[ModelParamsConfig, Dict[str, Union[int, float, bool]]],
        min_key: int,
        max_key: int,
        embedding_dim: int,
        num_features: int,
        config: Config,
    ) -> None:
        """
        Set model fields.

        This function sets model fields.

        Args:
            self ("LSTM"): Current class instance.
            params (Union[ModelParamsConfig, Dict[str, Union[int, float, bool]]]):
                Model parameters.
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
        debug(f"MC Dropout set to default: {self.mc_dropout}")

        # Set number of keys
        self.num_keys = max_key - min_key + 1
        debug(f"Number of keys for model: {self.num_keys}")

        # Set model input size
        self.input_size = num_features + embedding_dim
        debug(f"Input size for model: {self.input_size}")

        # Set number of features
        self.num_features = num_features
        debug(f"Number of features for model: {self.num_features}")

        # Set embedding with specified dimension
        self.embedding_dim = embedding_dim
        debug(f"Embedding dimension for model: {self.embedding_dim}")

        debug("Model fields set")

    def _set_layers(self: "LSTM") -> None:
        """
        Instantiate all the layers of the model.

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
            debug(f"Embedding layer instantiated (dim={self.embedding_dim})")

            # Instantiate dropout layer
            self.mc_dropout_layer = nn.Dropout(p=self.dropout)
            debug(f"Dropout layer instantiated (p={self.dropout})")

            # Instantiate fully connected layer
            self.fc = nn.Linear(self.hidden_size, self.num_keys)
            debug(
                f"Fully connected layer instantiated "
                f"(in_features={self.hidden_size}, out_features={self.num_keys})"
            )

            debug("Model layers set")
        except (TypeError, ValueError, AttributeError) as e:
            msg = "Failed to set model layers"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def __init__(
        self: "LSTM",
        params: Union[ModelParamsConfig, Dict[str, Union[int, float, bool]]],
        min_key: int,
        max_key: int,
        embedding_dim: int,
        num_features: int,
        config: Optional[Config],
    ) -> None:
        """
        Initialize the model.

        This function initializes the model, specifically:
        sets all fields, instantiates the class, as well as all its layers.

        Args:
            self ("LSTM"): Current class instance.
            params (Union[ModelParamsConfig, Dict[str, Union[int, float, bool]]]):
                Model parameters.
            min_key (int): Maximum key.
            max_key (int): Minimum key.
            embedding_dim (int): Embedding dimension for keys.
            num_features (int): Number of features for the model.
            config (Optional[Config]): Configuration object.

        Returns:
            None
        """
        super(LSTM, self).__init__()

        # Set model fields
        self._set_fields(
            params, min_key, max_key, embedding_dim, num_features, config
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
        debug(
            "LSTM model settings:\n"
            + f"input_size={self.input_size}\n"
            + f"hidden_size={self.hidden_size}\n"
            + f"num_layers={self.num_layers}\n"
            + f"bias={self.bias}\n"
            + f"batch_first={self.batch_first}\n"
            + f"dropout={self.dropout}\n"
            + f"bidirectional={self.bidirectional}\n"
            + f"proj_size={self.proj_size}"
        )

        # Set all model layers
        self._set_layers()

        debug("Model initialized")

    def _build_model_input(
        self: "LSTM", x_features: torch.Tensor, x_keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Build and return the model input.

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
            debug(
                f"Features shape: {x_features.shape}, "
                f"keys shape: {x_keys.shape}"
            )

            # Get the device of embedding layer
            device = self.embedding.weight.device
            debug(f"Embedding layer device: {device}")

            # Ensure keys are on the same device
            # as the embedding layer
            x_features = move_to_device(x_features, device)
            x_keys = move_to_device(x_keys, device)

            # Perform embedding
            embedded_keys = self.embedding(x_keys)
            debug(f"Embedded keys shape: {embedded_keys.shape}")

            # Concatenate features with embedded keys
            x = torch.cat((x_features, embedded_keys), dim=-1)
            debug(f"Model input shape: {x.shape}")

            debug("Model input retrieved")

            return x
        except (AttributeError, TypeError, RuntimeError) as e:
            msg = "Failed to build model input"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def forward(
        self: "LSTM", x_features: torch.Tensor, x_keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform a forward pass through the model.

        This function builds the model input by concatenating features
        with embedded keys, passes it through the model, optionally
        applies MC dropout, and finally computes logits using a linear layer.

        Args:
            x_features (torch.Tensor): Input features tensor for the current batch.
            x_keys (torch.Tensor): Input keys tensor for the current batch.

        Returns:
            torch.Tensor: Logits computed by the model.
        """
        try:
            debug(
                f"Forward pass with features"
                f" shape: {x_features.shape}, "
                f"keys shape: {x_keys.shape}"
            )

            # Build model input for current batch
            x = self._build_model_input(x_features, x_keys)

            # Pass the input through the model
            output, _ = self.lstm(x)
            debug(f"Model output shape: {output.shape}")

            # Apply MC dropout if enabled
            if self.mc_dropout:
                output = self.mc_dropout_layer(output)
                debug(f"(After MC dropout) Model output shape: {output.shape}")

            # Compute logits from last step
            logits = self.fc(output[:, -1, :])
            debug(f"Logits shape: {logits.shape}")

            debug("Model forward pass completed")

            return logits
        except (IndexError, RuntimeError) as e:
            msg = "Failed to compute model forward pass"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e
