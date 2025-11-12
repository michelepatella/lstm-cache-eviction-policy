"""access_logs_dataset.py

Dataset class for access logs compatible with PyTorch.

This module provides the `AccessLogsDataset` class, which manages sequential
access logs for model training and evaluation. It handles loading, preprocessing,
splitting, sequence extraction, and feature/target tensor conversion for PyTorch
models.

Classes:
    AccessLogsDataset(Dataset):
        PyTorch-compatible dataset class for sequential access logs.
"""

import torch
from torch.utils.data import Dataset

from components.const import (
    DATASET_PROCESSED_COLUMNS,
    DATASET_PROCESSED_FEATURE_COLUMNS,
    DATASET_TARGET_COLUMN_SHIFT,
    LIST_LAST_IDX,
    TORCH_DTYPE,
)
from components.dataset.columns.shifter import (
    shift_dataset_column,
)
from components.dataset.io.loader import load_dataset
from components.dataset.io.locator import get_dataset_abs_path
from components.dataset.rows.calculations.effective_rows_calculator import (
    calculate_effective_dataset_rows,
)
from components.dataset.rows.extractions.sliding_window_extractor import (
    extract_sliding_window_dataset_rows,
)
from components.dataset.splits.data_splitter import split_dataset_data
from components.dataset.splits.index.calculator import (
    calculate_dataset_split_index,
)
from components.logs.levels.debug_logger import debug
from components.logs.levels.error_logger import error
from const import DATASET_TRAINING_SPLIT_TYPE
from pipeline.config.pydantic.pipeline_config import PipelineConfig


class AccessLogsDataset(Dataset):
    """Dataset class for access logs, compatible with PyTorch.

    This class manages sequential access logs for a PyTorch models.
    It handles loading, preprocessing, splitting, and provides sequences
    of features and target keys for model training and evaluation.

    Attributes:
        data (pd.DataFrame): Full dataset stored internally.
        features (list[str]): List of feature column names.
        target (str): Target column name (usually the last column).
        seq_len (int): Sequence length used for model sequences.
    """

    def _split_dataset(
        self: "AccessLogsDataset",
        split_type: str,
        split_perc: float,
    ) -> None:
        """Split the dataset based on the requested dataset type.

        This function splits the dataset into either training
        or testing portions according to the train percentage
        provided.

        Args:
            self (AccessLogsDataset): Instance of AccessLogsDataset.
            split_type (str): The type of split to apply.
            split_perc (float): The split percentage.

        Returns:
            None
        """
        # Calculate dataset split index
        # based on split percentage
        dataset_split_idx = calculate_dataset_split_index(
            len(self.data),
            split_perc,
        )

        # Split dataset data according
        # to the calculated index
        self.data = split_dataset_data(
            self.data,
            dataset_split_idx,
            (split_type == DATASET_TRAINING_SPLIT_TYPE),
        )

    def _set_fields(
        self: "AccessLogsDataset",
        pipeline_config: PipelineConfig,
    ) -> None:
        """Set the feature, target, and sequence length fields of the dataset.

        This function sets the dataset columns, identifies feature columns
        and target colum, and retrieves the sequence length.

        Args:
            self (AccessLogsDataset): Instance of AccessLogsDataset.
            pipeline_config (PipelineConfig): Configuration object.

        Returns:
            None
        """
        # Set features and target of dataset
        self.features, self.target = (
            DATASET_PROCESSED_FEATURE_COLUMNS,
            DATASET_PROCESSED_COLUMNS[LIST_LAST_IDX],
        )

        # Set sequence length
        self.seq_len = pipeline_config.model.sequence.length

        debug(
            "Dataset fields setting executed",
            extra={
                "feature_columns": self.features,
                "target_column": self.target,
                "sequence_length": self.seq_len,
                "context": "AccessLogsDataset",
            },
        )

    def __init__(
        self: "AccessLogsDataset",
        dataset_type: str,
        split_type: str,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Initialize the AccessLogsDataset class.

        This function initializes the AccessLogsDataset class,
        for a given dataset type, by loading dataset, preparing
        and setting data, and splitting the dataset.

        Args:
            self (AccessLogsDataset): AccessLogsDataset class.
            dataset_type (str): The dataset type requested to
                                be created.
            split_type (str): The split type requested.
            pipeline_config (PipelineConfig): Configuration object.

        Returns:
            None
        """
        # Prepare configuration
        data_mode = pipeline_config.data.general.mode
        training_split = pipeline_config.dataset.splits.training

        # Retrieve path to load dataset from
        dataset_path = get_dataset_abs_path(
            dataset_type,
            data_mode,
        )

        # Load the dataset
        df = load_dataset(dataset_path)

        # Set data
        self.data = df.copy()

        # Split the dataset
        self._split_dataset(split_type, training_split)

        # Set the fields of the dataset
        self._set_fields(pipeline_config)

        # Shift target by -1
        shift_dataset_column(
            self.data,
            self.target,
            DATASET_TARGET_COLUMN_SHIFT,
        )

        debug(
            "AccessLogsDataset initialization executed",
            extra={
                "dataset_type": dataset_type,
                "rows_num": len(self.data),
                "columns_num": len(self.data.columns),
                "feature_columns": self.features,
                "target_column": self.target,
                "sequence_length": self.seq_len,
                "context": "AccessLogsDataset",
            },
        )

    def __len__(self: "AccessLogsDataset") -> int:
        """Return the effective length of the access logs dataset.

        This method calculates the number of sequences
        available in the dataset based on the sequence length.

        Args:
            self (AccessLogsDataset): AccessLogsDataset class.

        Returns:
            int: Number of available sequences
                 in the dataset.
        """
        return calculate_effective_dataset_rows(self.data, self.seq_len)

    def __getitem__(
        self: "AccessLogsDataset",
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve a sequence of features and the next target key.

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
            RuntimeError: If an error occurs while retrieving a dataset item:
                * Requested index goes out of bounds (IndexError).
                * Feature or target columns are missing (KeyError).
                * Conversion to int/float fails (TypeError, ValueError).
                * Not enough data to extract sequence (custom RuntimeError).
        """
        try:
            # Extract data sequence
            seq_data = extract_sliding_window_dataset_rows(
                self.data,
                idx + self.seq_len - 1,
                self.seq_len,
            )

            # Check whether there is not
            # enough sequence data
            if seq_data is None:
                msg = "Not enough data to extract sequence"
                error(
                    msg,
                    extra={
                        "index_requested": idx,
                        "sequence_length": self.seq_len,
                        "rows_available": len(self.data),
                        "context": "AccessLogsDataset",
                    },
                )
                raise RuntimeError(msg)

            # Convert features to float tensor and keys
            # to long tensor
            x_features = torch.tensor(
                seq_data[self.features].values.astype(float),
                dtype=TORCH_DTYPE,
            )
            x_keys = torch.tensor(
                seq_data[self.target].values.astype(int),
                dtype=TORCH_DTYPE,
            )

            # Get the next target key
            target_row = self.data.iloc[idx + self.seq_len]
            y_key = torch.tensor(
                int(target_row[self.target]),
                dtype=TORCH_DTYPE,
            )
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as e:
            msg = "Retrieving dataset item failed"
            error(
                msg,
                extra={
                    "exception": str(e),
                    "index_requested": idx,
                    "sequence_length": self.seq_len,
                    "context": "AccessLogsDataset",
                },
            )
            raise RuntimeError(msg) from e

        return x_features, x_keys, y_key
