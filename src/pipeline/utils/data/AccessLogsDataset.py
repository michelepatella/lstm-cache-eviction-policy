from typing import Tuple, Type

import pandas as pd
import torch
from torch.utils.data import Dataset

from const import DATASET_PREPROCESSED_TYPE, TRAINING_SPLIT_TYPE
from pipeline.config.classes.Config import Config
from pipeline.utils.data.dataset.dataset_loader import load_dataset
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.error_logger import error
from pipeline.utils.logs.levels.info_logger import info


class AccessLogsDataset(Dataset):
    """
    Dataset class for access logs, compatible with PyTorch.

    This class manages sequential access logs for LSTM model. It handles
    loading, preprocessing, and splitting of the dataset into training
    and testing sets. Each dataset instance provides feature sequences
    and target keys for sequence modeling.
    """

    def _split_dataset(
        self: "AccessLogsDataset", dataset_type: str, config: Config
    ) -> None:
        """
        Split the dataset based on the requested dataset type.

        This function splits the dataset into either training
        or testing portions according to the train percentage
        defined in the configuration.

        Parameters:
            self (AccessLogsDataset): Instance of AccessLogsDataset.
            dataset_type (str): The dataset type to split (e.g., "training").
            config (Config): Configuration object.

        Returns:
            None

        Raises:
            RuntimeError: If an error occurs during dataset splitting, e.g.:
                * self.data is not a Pandas DataFrame.
                * self.data length or train % is of wrong type.
                * self.data length or train % cannot be converted to int.
                * the split index is out of bounds for self.data.
        """
        debug(f"Dataset splitting type: {dataset_type}")

        try:
            # Calculate split index based on the train %
            train_perc = config.data.dataset.split.training
            dataset_split_idx = int(len(self.data) * train_perc)

            debug(f"Dataset split index calculated: {dataset_split_idx}")
        except (AttributeError, TypeError, ValueError) as e:
            msg = "Failed to calculate dataset split index"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

        try:
            # Perform dataset split
            if dataset_type == TRAINING_SPLIT_TYPE:
                # Training -> Take the first dataset
                # split index elements
                self.data = self.data[:dataset_split_idx]
            else:
                # Testing (Otherwise) -> Take the last dataset
                # split index elements
                self.data = self.data[dataset_split_idx:]

            debug(f"Dataset shape after splitting: {self.data.shape}")
        except (TypeError, IndexError, AttributeError) as e:
            msg = "Failed to split dataset"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

        info("Dataset split")

    def _set_fields(
        self: "AccessLogsDataset", data: "pd.DataFrame", config: Config
    ) -> None:
        """
        Set the feature, target, and sequence length fields of the dataset.

        This function sets the dataset columns, identifies feature columns
        and target column (assuming target is the last column), and retrieves
        the sequence length from the configuration.

        Parameters:
            self (AccessLogsDataset): Instance of AccessLogsDataset.
            data (pd.DataFrame): Dataset from which fields are extracted.
            config (Config): Configuration object.

        Returns:
            None

        Raises:
            RuntimeError: If an error occurs while
                          setting dataset fields, e.g.:
                * data is not a Pandas DataFrame
                * data columns or sequence length are of wrong type
                * data does not contain any columns
        """
        try:
            # Extract column names
            self.columns = data.columns.tolist()

            debug(f"Dataset columns extracted: {self.columns}")
        except (AttributeError, TypeError) as e:
            msg = "Failed to extract dataset columns"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

        try:
            # Identify features and target
            self.features = self.columns[
                :-1
            ]  # All the columns except the last one
            self.target = self.columns[-1]  # Only the last one

            # Retrieve sequence length
            self.seq_len = config.data.sequence.length

            debug(f"Feature dataset columns: {self.features}")
            debug(f"Target dataset column: {self.target}")
            debug(f"Sequence length: {self.seq_len}")
        except (IndexError, AttributeError, TypeError) as e:
            msg = "Failed to set dataset fields"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

        info("Dataset fields set")

    def __init__(
        self: "AccessLogsDataset", dataset_type: str, config: Config
    ) -> None:
        """
        Initialize the AccessLogsDataset class.

        This function initializes the AccessLogsDataset class,
        for a given dataset type, by loading dataset, preparing
        and setting data, and splitting the dataset.

        Parameters:
            self (AccessLogsDataset): AccessLogsDataset class.
            dataset_type (str): The dataset type requested to
                                be created (e.g. "training").
            config (Config): Configuration object.

        Returns:
            None

        Raises:
            RuntimeError: If an error occurs while
                          initializing the dataset, e.g.:
                * dataset loaded is empty
                  a value on the target column
                  cannot be converted to integer
                * a value on the target column is not
                  compatible with astype(int)
                * dataset loaded is not a Pandas DataFrame
                  or missing columns attribute
        """
        debug(f"AccessLogsDataset type to be initialized: {dataset_type}")

        # Load the dataset
        df = load_dataset(DATASET_PREPROCESSED_TYPE, config)

        try:
            # Shift target column (requests),
            # ensuring requests keys are in range
            # (min_key - 1, max_key - 1)
            df[df.columns[-1]] = df[df.columns[-1]].astype(int) - 1
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            msg = "Failed to initialize AccessLogsDataset"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

        # Set data
        self.data = df.copy()

        # Split the dataset to assign data properly
        self._split_dataset(dataset_type, config)

        # Set the fields of the dataset
        self._set_fields(self.data, config)

        info("AccessLogsDataset initialized")

    def __len__(self: "AccessLogsDataset") -> int:
        """
        Return the effective length of the access logs dataset.

        This method calculates the number of sequences
        available in the dataset based on the sequence length.

        Parameters:
            self (AccessLogsDataset): AccessLogsDataset class.

        Returns:
            int: Number of available sequences
                 in the dataset.

        Raises:
            RuntimeError: If an error occurs while
                          calculating dataset length, e.g.:
                * self.data or self.seq_len is
                  not properly set
                * self.data is not iterable or sequence
                  length is not an integer
                * calculated length is negative
        """
        try:
            # Calculate the dataset length by
            # removing from available data the
            # sequence length
            dataset_length = len(self.data) - self.seq_len

            debug(f"Calculated dataset length: {dataset_length}")

            # Check whether the dataset
            # length is negative
            if dataset_length < 0:
                msg = "Calculated dataset length is negative"
                error("%s", msg)
                raise RuntimeError(msg)

            return dataset_length
        except (AttributeError, TypeError, ValueError) as e:
            msg = "Failed to get dataset length"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

    def __getitem__(
        self: "AccessLogsDataset", idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a sequence of features and the next target key.

        This method extracts a feature sequence of length
        equal to those of sequence and the corresponding
        next key in the dataset.

        Parameters:
            self (AccessLogsDataset): AccessLogsDataset class.
            idx (int): Index of the starting row
                       for the sequence.

        Returns:
            Tuple[
                torch.Tensor, torch.Tensor, torch.Tensor
            ]: Tuple containing a tensor of
               numerical features, a tensor
               of keys in the  sequence,
               and a tensor for the next key.

        Raises:
            RuntimeError: If an error occurs while
                          retrieving a dataset item, e.g.:
                * The requested index goes out of bounds
                * Feature or target columns are missing
                * Data types in features/target are invalid
                * Conversion to int/float fails
                * self.data, self.features, or self.target
                  are not set properly
        """
        debug(f"Index of item to be retrieved: {idx}")

        try:
            # Extract feature sequence
            seq_data = self.data.iloc[idx:idx + self.seq_len]

            # Convert features to float tensor
            x_features = torch.tensor(
                seq_data[self.features].values.astype(float), dtype=torch.float
            )

            # Convert keys to long tensor
            x_keys = torch.tensor(
                seq_data[self.target].values.astype(int), dtype=torch.long
            )

            # Get next target key
            target_row = self.data.iloc[idx + self.seq_len]
            y_key = torch.tensor(
                int(target_row[self.target]), dtype=torch.long
            )

            debug(f"Feature tensor shape: {x_features.shape}")
            debug(f"Key tensor shape: {x_keys.shape}")
            debug(f"Target key: {y_key.item()}")
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
        ) as e:
            msg = "Failed to retrieve dataset item"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

        return x_features, x_keys, y_key

    @classmethod
    def from_dataframe(
        cls: "Type[AccessLogsDataset]", df: "pd.DataFrame", config: Config
    ) -> "AccessLogsDataset":
        """
        Instantiate AccessLogsDataset from an existing dataframe.

        This method creates an instance of AccessLogsDataset using a
        preloaded dataframe, copying its content and setting the
        necessary fields (features, target, sequence length).

        Parameters:
            cls (AccessLogsDataset): AccessLogsDataset class.
            df (pd.DataFrame): Preloaded dataframe containing dataset.
            config (Config): Configuration object.

        Returns:
            AccessLogsDataset: Initialized dataset instance.

        Raises:
            RuntimeError: If an error occurs while
                          instantiating from dataframe, e.g.:
                * The dataframe does not have expected attributes
                * The provided object is not a Pandas DataFrame
                * The dataframe does not contain any columns
        """
        debug(
            f"Dataframe shape to be instantiated"
            f" as AccessLogsDataset: {df.shape}"
        )

        try:
            # Create a new dataset instance
            instance = cls.__new__(cls)

            # Copy the dataframe to the instance
            instance.data = df.copy()

            # Set dataset fields
            instance._set_fields(df, config)
        except (AttributeError, TypeError, IndexError) as e:
            msg = "Failed to instantiate AccessLogsDataset from dataframe"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

        info("AccessLogsDataset instantiated from dataframe")

        return instance
