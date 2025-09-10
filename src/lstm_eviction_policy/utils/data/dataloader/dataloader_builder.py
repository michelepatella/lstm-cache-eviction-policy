from torch.utils.data import DataLoader
from utils.logs.log_utils import debug, info


def create_data_loader(dataset, batch_size, shuffle=False):
    """
    Method to create data loader from a dataset.
    :param dataset: The dataset to load.
    :param batch_size: The batch size to use.
    :param shuffle: Whether to shuffle the dataset.
    :return: The data loader.
    """
    # initial message
    info("🔄 Data loader creation started...")

    # debugging
    debug(f"⚙️ Batch size: {batch_size}.")
    debug(f"⚙️ Shuffle: {shuffle}.")

    try:
        # define the loader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # show a successful message
    info("🟢 Data loader created.")

    return loader
