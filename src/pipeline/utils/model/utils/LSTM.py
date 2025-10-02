import torch
import torch.nn as nn

from const import LSTM_PARAMETERS, MC_DROPOUT_DEFAULT
from pipeline.config.classes.Config import Config
from pipeline.config.classes.ModelConfig import ModelParamsConfig
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


class LSTM(nn.Module):
    def _set_fields(
        self: "LSTM", params: ModelParamsConfig, config: Config
    ) -> None:
        """
        Set LSTM model fields.

        This function sets LSTM model fields using
        provided parameters or configuration.

        Parameters:
            self ("LSTM"): Current class instance.
            params (ModelParamsConfig): Parameters specified for the model.
            config (Config): Configuration object.

        Returns:
            None

        Raises:
            RuntimeError: If LSTM fields assignment fails, e.g.:
                * Invalid values for embedding dimension or number of keys.
        """
        try:
            # Prepare configuration
            model_params = config.model.params
            max_key = config.data.general.keys.max
            min_key = config.data.general.keys.min
            embedding_dim = config.data.sequence.embedding.dimension
            num_features = config.model.general.features.count

            # For each required parameter
            for param in LSTM_PARAMETERS:
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
                        f"'{param}' not found or None, using config value: {config_value}"
                    )

                    # Set the value specified by configuration
                    setattr(self, param, config_value)

            # Set number of keys
            self.num_keys = max_key - min_key + 1

            debug(f"Number of keys for LSTM model: {self.num_keys}")

            # Set model input size
            self.input_size = num_features + embedding_dim

            debug(f"Input size for LSTM model: {self.input_size}")

            # Set number of features
            self.num_features = num_features

            debug(f"Number of features for LSTM model: {self.num_features}")

            # Apply embedding with specified dimension
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(self.num_keys, self.embedding_dim)

            debug(
                f"Embedding layer for LSTM model instantiated with dimension: {self.embedding_dim}"
            )

            info("LSTM fields set")
        except (TypeError, ValueError) as e:
            msg = "Failed to set LSTM fields"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def __init__(
        self: "LSTM", params: ModelParamsConfig, config: Config
    ) -> None:
        """
        Initialize the LSTM model.

        This function initializes the LSTM model, specifically:
        set MC dropout default value, set all LSTM fields, instantiate
        the LSTM class, as well as a dropout and a fully-connected (linear)
        layer.

        Parameters:
            self ("LSTM"): Current class instance.
            params (ModelParamsConfig: Model parameters.
            config (Config): Configuration object.

        Returns:
            None
        """
        super(LSTM, self).__init__()

        # Set MC dropout flag
        self.mc_dropout = MC_DROPOUT_DEFAULT

        debug(f"MC dropout set to default: {self.mc_dropout}")

        # Set model fields
        self._set_fields(params, config)

        # Instantiate the LSTM layer
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

        # Instantiate a dropout layer
        self.mc_dropout_layer = nn.Dropout(p=self.dropout)

        debug(f"Dropout layer instantiated (p={self.dropout})")

        # Instantiate a fully connected layer
        self.fc = nn.Linear(self.hidden_size, self.num_keys)

        debug(
            f"Fully connected layer instantiated (in_features={self.hidden_size},"
            f" out_features={self.num_keys})"
        )

        info("LSTM model initialized")

    def _build_lstm_input(
        self: "LSTM", x_features: torch.Tensor, x_keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Build and return the LSTM input.

        This function builds and returns the LSTM input
        by concatenating features with embedded keys.

        Parameters:
            x_features (torch.Tensor): Feature tensor for the current batch.
            x_keys (torch.Tensor): Keys tensor for the current batch.

        Returns:
            torch.Tensor: Concatenated input tensor ready for LSTM.

        Raises:
            RuntimeError: If an error occurs while building the LSTM input, e.g.:
                * Keys contains out-of-range values for embedding lookup.
                * Features or embedded keys are of incompatible shape for concatenation.
                * Features or keys are of invalid type (non-tensor).
        """
        try:
            debug(
                f"Features shape: {x_features.shape}, keys shape: {x_keys.shape}"
            )

            # Pass keys through the embedding layer
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

        Parameters:
            x_features (torch.Tensor): Input features tensor for the current batch.
            x_keys (torch.Tensor): Input keys tensor for the current batch.

        Returns:
            torch.Tensor: Logits computed by the LSTM model.

        Raises:
            RuntimeError: If an error occurs during the forward pass, e.g.:
                * LSTM input construction fails due to incompatible shapes or types.
                * LSTM computation fails due to invalid input dimensions.
                * MC dropout layer application fails.
                * Fully-connected layer computation fails due to invalid tensor shape.
                * Indexing fails when extracting the last timestep of LSTM output.
        """
        try:
            debug(
                f"LSTM forward pass with features shape: {x_features.shape}, keys shape: {x_keys.shape}"
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
