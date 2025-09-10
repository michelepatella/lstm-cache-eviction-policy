import pandas as pd
from utils.data.dataset.dataset_utils import get_dataset_path_type
from utils.logs.log_utils import debug, info


def load_dataset(config_settings):
    """
    Method to load the dataset.
    :param config_settings: The configuration settings.
    :return: The dataset read.
    """
    # initial message
    info("🔄 Dataset loading started...")

    # get the dataset path
    dataset_path = get_dataset_path_type(config_settings)

    # debugging
    debug(f"⚙️ Path of the dataset to be loaded: {dataset_path}.")

    try:
        # load the dataset
        df = pd.read_csv(dataset_path)
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # debugging
    debug(f"⚙️ Shape of the dataset loaded: {df.shape}.")

    # show a successful message
    info("🟢 Dataset loaded.")

    return df
