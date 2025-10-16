from typing import Tuple, Type

import pandas as pd
import torch
from torch.utils.data import Dataset

from pipeline.config.classes.Config import Config
from pipeline.const import TRAINING_SPLIT_TYPE
from utils.dataset.columns.extractor import extract_dataset_columns
from utils.dataset.extraction.features_target_from_columns_extractor import (
    extract_features_target_from_dataset_columns,
)
from utils.dataset.io.loader import load_dataset
from utils.dataset.io.locator import get_dataset_abs_path
from utils.dataset.splitting.data_splitter import split_dataset_data
from utils.dataset.splitting.index_calculator import (
    calculate_dataset_split_index,
)
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.error_logger import error
from utils.logs.levels.info_logger import info


def add_dataset_column(df, column_name, column_values):
    pass


class AccessLogsDataset(Dataset):
    """
    Dataset class for access logs, compatible with PyTorch.

    This class manages sequential access logs for a PyTorch models.
    It handles loading, preprocessing, splitting, and provides sequences
    of features and target keys for model training and evaluation.

    Attributes:
        data (pd.DataFrame): Full dataset stored internally.
        columns (List[str]): List of column names in the dataset.
        features (List[str]): List of feature column names.
        target (str): Target column name (usually the last column).
        seq_len (int): Sequence length used for LSTM sequences.
    """

    def _split_dataset(
        self: "AccessLogsDataset", dataset_type: str, split_perc: int
    ) -> None:
        """
        Split the dataset based on the requested dataset type.

        This function splits the dataset into either training
        or testing portions according to the train percentage
        provided.

        Args:
            self (AccessLogsDataset): Instance of AccessLogsDataset.
            dataset_type (str): The dataset type to split.
            split_perc (int): The split percentage.

        Returns:
            None
        """
        debug(f"Dataset splitting type: {dataset_type}")

        # Calculate dataset split index
        # based on split percentage
        dataset_split_idx = calculate_dataset_split_index(
            len(self.data), split_perc
        )

        # Split dataset data according
        # to the calculated index
        self.data = split_dataset_data(
            self.data, dataset_split_idx, (dataset_type == TRAINING_SPLIT_TYPE)
        )

        info("Dataset split")

    def _set_fields(
        self: "AccessLogsDataset", data: "pd.DataFrame", config: Config
    ) -> None:
        """
        Set the feature, target, and sequence length fields of the dataset.

        This function sets the dataset columns, identifies feature columns
        and target colum, and retrieves the sequence length.

        Args:
            self (AccessLogsDataset): Instance of AccessLogsDataset.
            data (pd.DataFrame): Dataset from which fields are extracted.
            config (Config): Configuration object.

        Returns:
            None
        """
        # Set column names extracted from data
        self.columns = extract_dataset_columns(data)

        # Set features and target extracted from columns
        self.features, self.target = (
            extract_features_target_from_dataset_columns(self.columns)
        )

        # Set sequence length
        self.seq_len = config.model.sequence.length

        info("Dataset fields set")

    def __init__(
        self: "AccessLogsDataset", dataset_type: str, config: Config
    ) -> None:
        """
        Initialize the AccessLogsDataset class.

        This function initializes the AccessLogsDataset class,
        for a given dataset type, by loading dataset, preparing
        and setting data, and splitting the dataset.

        Args:
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

        # Prepare configuration
        data_distribution_mode = config.data.generation.mode
        training_split = config.dataset.split.training

        # Retrieve path to load dataset from
        dataset_path = get_dataset_abs_path(
            dataset_type, data_distribution_mode
        )

        # Load the dataset
        df = load_dataset(dataset_path)

        try:
            # Shift target column (requests),
            # ensuring requests keys are in range
            # (min_key - 1, max_key - 1)
            column_name = df.columns[-1]
            column_values = df[column_name].astype(int) - 1
            add_dataset_column(df, column_name, column_values)
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            msg = "Failed to initialize AccessLogsDataset"
            error("%s: %s", msg, e)
            raise RuntimeError(msg) from e

        # Set data
        self.data = df.copy()

        # Split the dataset to assign data properly
        self._split_dataset(dataset_type, training_split)

        # Set the fields of the dataset
        self._set_fields(self.data, config)

        info("AccessLogsDataset initialized")

    def __len__(self: "AccessLogsDataset") -> int:
        """
        Return the effective length of the access logs dataset.

        This method calculates the number of sequences
        available in the dataset based on the sequence length.

        Args:
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

        Args:
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
            seq_data = self.data.iloc[idx : idx + self.seq_len]

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

        Args:
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
