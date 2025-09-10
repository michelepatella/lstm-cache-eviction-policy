from utils.logs.log_utils import debug, info


def get_dataset_path_type(config_settings):
    """
    Method to get the dataset path and type from config file.
    :param config_settings: The configuration settings.
    :return: The dataset path.
    """
    # initial message
    info("🔄 Dataset path and type retrieval started...")

    # debugging
    debug(
        f"⚙️ Dataset distribution type from config: "
        f"{config_settings.distribution_type}."
    )

    # keep track of the dataset path
    if config_settings.distribution_type == "static":
        dataset_path = config_settings.static_save_path
    else:
        dataset_path = config_settings.dynamic_save_path

    # debugging
    debug(f"⚙️ Dataset path found: {dataset_path}.")

    # show a successful message
    info("🟢 Dataset path and type retrieved.")

    return dataset_path
