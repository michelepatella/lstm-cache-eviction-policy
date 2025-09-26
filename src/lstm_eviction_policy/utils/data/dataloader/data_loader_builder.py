from torch.utils.data import DataLoader

from lstm_eviction_policy.utils.data.AccessLogsDataset import AccessLogsDataset
from lstm_eviction_policy.utils.logs.log_utils import debug, error, info


def create_data_loader(
    dataset: AccessLogsDataset, batch_size: int, shuffle: bool
) -> DataLoader:
    """
    Create a data loader for the given dataset.

    This function creates a data loader for a given dataset,
    applying specified settings including batch size and shuffling.

    Parameters:
        dataset (AccessLogsDataset): The dataset instance to create the
                                     data loader for.
        batch_size (int): The batch size to use for the data loader.
        shuffle (bool): Whether to shuffle the data loader.

    Returns:
        DataLoader: The data loader created.

    Raises:
        TypeError: If dataset object is not compatible
                   with torch.utils.data.Dataset, or batch size
                   or shuffle are not of correct type.
        ValueError: If one or more parameters (e.g., batch size)
                    are not valid.
    """
    debug(f"Batch size for the data loader to be created: {batch_size}")
    debug(f"Shuffle for the data loader to be created: {shuffle}")

    try:
        # Instantiate the data loader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    except (TypeError, ValueError) as e:
        msg = "Failed to create data loader"
        error("%s: %s", msg, e)
        raise RuntimeError(msg) from e

    info(f"Data loader created")

    return loader
