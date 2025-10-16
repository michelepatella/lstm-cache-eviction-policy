from typing import Dict

import torch
import torch.nn as nn

from pipeline.config.classes.Config import Config
from pipeline.config.classes.ModelConfig import ModelParamsConfig
from pipeline.const import LSTM_PARAM_NAMES, MC_DROPOUT_DEFAULT
from utils.device.mover import move_to_device
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


class LSTM(torch.nn.Module):
    """
    LSTM model for cache eviction prediction.

    This class implements an LSTM-based model that predicts which key
    to evict from the cache.

    Attributes:
        mc_dropout (bool): Flag indicating if MC dropout is enabled.
        num_keys (int): Number of unique keys to embed.
        input_size (int): Size of LSTM input (features + embedding dim).
        num_features (int): Number of features in input data.
        embedding_dim (int): Dimensionality of key embeddings.
        embedding (nn.Embedding): Embedding layer for keys.
        lstm (nn.LSTM): PyTorch LSTM module.
        mc_dropout_layer (nn.Dropout): Dropout layer for MC dropout.
        fc (nn.Linear): Fully connected layer producing logits.
    """

    def _set_params(
            self: "LSTM",
            params: ModelParamsConfig | Dict[str, int | float | bool],
            config: Config | None,
            param_names: list[str] = LSTM_PARAM_NAMES
    ) -> None:
        """
        Set model parameters.

        This function sets model parameters by using values provided directly,
        or falling back to configuration values if not provided.

        Args:
            self ("LSTM"): Current model instance.
            params (ModelParamsConfig | Dict[str, int | float | bool]):
                Dictionary of parameters explicitly passed.
            config (Config | None): Configuration object.
            param_names (list[str]): List of parameter names to set.

        Raises:
            RuntimeError: If an error occurs while setting model parameters e.g.:
                * Provided params dictionary is invalid.
                * Configuration object is missing or malformed.
                * Attribute assignment on the model instance fails.
        """
        try:
            # Prepare configuration model params
            # if a configuration object is passed
            model_params = None
            if config is not None:
                model_params = config.model.params

            # For each required parameter
            for param in param_names:
                # Check whether the required parameter
                # has been directly passed to the model
                if param in params and params[param] is not None:
                    debug(f"'{param}' found: {params[param]}")

                    # Set the specified parameter value
                    setattr(self, param, params[param])
                else:
                    # Otherwise, retrieve parameter value from
                    # configuration
                    config_value = getattr(model_params, param)

                    debug(
                        f"'{param}' not found or None,"
                        f" using config value: {config_value}"
                    )

                    # Set the value specified by configuration
                    setattr(self, param, config_value)

            info("Model parameters set")
        except (AttributeError, TypeError, ValueError) as e:
            msg = "Failed to set model parameters"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def _set_fields(
        self: "LSTM",
        params: ModelParamsConfig | Dict[str, int | float | bool],
        min_key: int,
        max_key: int,
        embedding_dim: int,
        num_features: int,
        config: Config,
    ) -> None:
        """
        Set LSTM model fields.

        This function sets LSTM model fields using
        provided parameters and/or configuration.

        Args:
            self ("LSTM"): Current class instance.
            params (ModelParamsConfig | Dict[str, int | float | bool]):
                Parameters specified for the model.
            min_key (int): Maximum key.
            max_key (int): Minimum key.
            embedding_dim (int): Embedding dimension.
            num_features (int): Number of features for the LSTM model.
            config (Config): Configuration object.

        Returns:
            None
        """
        # Set model params
        self._set_params(params, config)

        # Set MC dropout default flag
        self.mc_dropout = MC_DROPOUT_DEFAULT
        debug(f"MC dropout set to default: {self.mc_dropout}")

        # Set number of keys
        self.num_keys = max_key - min_key + 1
        debug(f"Number of keys for LSTM model: {self.num_keys}")

        # Set model input size
        self.input_size = num_features + embedding_dim
        debug(f"Input size for LSTM model: {self.input_size}")

        # Set number of features
        self.num_features = num_features
        debug(f"Number of features for LSTM model: {self.num_features}")

        # Set embedding with specified dimension
        self.embedding_dim = embedding_dim
        debug(f"Embedding dimension for LSTM model: {self.embedding_dim}")

        info("LSTM fields set")

    def _set_layers(self: "LSTM") -> None:
        """
        Instantiate all the layers of the LSTM model.

        This function creates and assigns the following layers to the model:
            - Embedding layer
            - Dropout layer
            - Fully connected (linear) layer

        Raises:
            RuntimeError: If an error occurs while setting the model layers e.g.:
                * Invalid input dimensions layers.
                * Attributes required for layer creation are missing or None.
        """
        try:
            # Instantiate embedding layer
            self.embedding = nn.Embedding(self.num_keys, self.embedding_dim)
            debug(f"Embedding layer instantiated with dimension: {self.embedding_dim}")

            # Instantiate dropout layer
            self.mc_dropout_layer = nn.Dropout(p=self.dropout)
            debug(f"Dropout layer instantiated (p={self.dropout})")

            # Instantiate fully connected layer
            self.fc = nn.Linear(self.hidden_size, self.num_keys)
            debug(f"Fully connected layer instantiated "
                  f"(in_features={self.hidden_size}, out_features={self.num_keys})")

            info("Model layers set")
        except (TypeError, ValueError, AttributeError) as e:
            msg = "Failed to set model layers"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def __init__(
        self: "LSTM",
        params: ModelParamsConfig | Dict[str, int | float | bool],
        min_key: int,
        max_key: int,
        embedding_dim: int,
        num_features: int,
        config: Config = None,
    ) -> None:
        """
        Initialize the LSTM model.

        This function initializes the LSTM model, specifically:
        sets all LSTM fields, instantiates the LSTM class,
        as well as all the model layers.

        Args:
            self ("LSTM"): Current class instance.
            params (ModelParamsConfig | Dict[str, int | float | bool]):
                Model parameters.
            min_key (int): Maximum key.
            max_key (int): Minimum key.
            embedding_dim (int): Embedding dimension.
            num_features (int): Number of features for the LSTM model.
            config (Config): Configuration object.

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

        info("LSTM model initialized")

    def _build_lstm_input(
        self: "LSTM", x_features: torch.Tensor, x_keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Build and return the LSTM input.

        This function builds and returns the LSTM input
        by concatenating features with embedded keys.

        Args:
            x_features (torch.Tensor): Feature tensor for the current batch.
            x_keys (torch.Tensor): Keys tensor for the current batch.

        Returns:
            torch.Tensor: Concatenated input
                          tensor ready for LSTM.

        Raises:
            RuntimeError: If an error occurs while
                          building the LSTM input, e.g.:
                * Keys contains out-of-range values
                  for embedding lookup.
                * Features or embedded keys are of
                  incompatible shape for concatenation.
                * Features or keys are of invalid type (non-tensor).
        """
        try:
            debug(
                f"Features shape: {x_features.shape}, "
                f"keys shape: {x_keys.shape}"
            )

            # Ensure keys are on the same device
            # as the embedding layer
            device = self.embedding.weight.device
            x_features = move_to_device(x_features, device)
            x_keys = move_to_device(x_keys, device)

            # Perform embedding
            embedded_keys = self.embedding(x_keys)
            debug(f"Embedded keys shape: {embedded_keys.shape}")

            # Concatenate features with embedded keys
            x = torch.cat((x_features, embedded_keys), dim=-1)

            debug(f"LSTM input shape: {x.shape}")

            info("LSTM input retrieved")

            return x
        except (IndexError, TypeError, RuntimeError) as e:
            msg = "Failed to build LSTM input"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def forward(
        self: "LSTM", x_features: torch.Tensor, x_keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform a forward pass through the LSTM model.

        This function builds the LSTM input by concatenating features
        with embedded keys, passes it through the LSTM, optionally
        applies MC dropout, and finally computes logits using a linear layer.

        Args:
            x_features (torch.Tensor): Input features tensor
                                       for the current batch.
            x_keys (torch.Tensor): Input keys tensor
                                   for the current batch.

        Returns:
            torch.Tensor: Logits computed by the LSTM model.

        Raises:
            RuntimeError: If an error occurs during
                          the forward pass, e.g.:
                * LSTM input construction fails due
                  to incompatible shapes or types.
                * LSTM computation fails due to
                  invalid input dimensions.
                * MC dropout layer application fails.
                * Fully-connected layer computation
                  fails due to invalid tensor shape.
                * Indexing fails when extracting the
                  last timestep of LSTM output.
        """
        try:
            debug(
                f"LSTM forward pass with features"
                f" shape: {x_features.shape}, "
                f"keys shape: {x_keys.shape}"
            )

            # Build LSTM input for current batch
            x = self._build_lstm_input(x_features, x_keys)

            # Pass the input through LSTM
            lstm_out, _ = self.lstm(x)
            debug(f"LSTM output shape: {lstm_out.shape}")

            # Apply MC dropout if enabled
            if self.mc_dropout:
                lstm_out = self.mc_dropout_layer(lstm_out)
                debug(
                    f"(After MC dropout) LSTM output shape: {lstm_out.shape}"
                )

            # Compute logits from last timestep
            logits = self.fc(lstm_out[:, -1, :])
            debug(f"Logits shape: {logits.shape}")

            info("LSTM forward pass completed")

            return logits
        except (IndexError, RuntimeError) as e:
            msg = "Failed to compute LSTM forward pass"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e
