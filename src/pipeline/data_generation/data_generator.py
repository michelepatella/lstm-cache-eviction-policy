from const import (
    DATA_DISTRIBUTION_STATIC_MODE,
    DATASET_RAW_TYPE,
    LOGS_DATA_GENERATION_PHASE,
    REQUEST_COLUMN,
    TIMESTAMP_COLUMN, PLOTS_SAVE_PATH, KEY_USAGE_SAVE_PATH_NAME, DAILY_PROFILE_SAVE_PATH_NAME,
    ZIPF_LOGLOG_SAVE_PATH_NAME,
)
from config.classes.Config import Config
from pipeline.data_generation.generation.requests.dynamic_generator import (
    generate_dynamic_requests,
)
from pipeline.data_generation.generation.requests.static_generator import (
    generate_static_requests,
)
from pipeline.data_generation.utils.dataframe_builder import create_dataframe
from pipeline.data_generation.visualization.plotting.daily_profile_plotter import (
    plot_daily_profile,
)
from pipeline.data_generation.visualization.plotting.key_usage_heatmap_plotter import (
    plot_key_usage_heatmap,
)
from pipeline.data_generation.visualization.plotting.zipf_loglog_plotter import (
    plot_zipf_loglog,
)
from pipeline.utils.dataset.saver import save_dataset
from utils.logs.initializer import logs_phase
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


def generate_data(config: Config) -> None:
    """
    Generate data according
    to a specified data distribution mode.

    This function generates data according
    to a specified data distribution mode, by
    orchestrating the generation of both access
    and temporal data patterns of requests. These
    patterns aim to reflect real-world data access
    patterns. Data generated — including a requested
    key and the corresponding timestamp in hours —
    is used to create a dataframe saved as CSV dataset next.
    Finally, data generated is validated by proper plots.

    Parameters:
        config (Config): Configuration object.

    Returns:
        None
    """
    # Set the new state
    logs_phase.set(LOGS_DATA_GENERATION_PHASE)

    # Prepare configuration
    data_distribution_mode = config.data.general.mode
    min_key = config.data.general.keys.min
    max_key = config.data.general.keys.max

    debug(f"Data distribution mode: {data_distribution_mode}")

    # Generate requests with corresponding timestamps,
    # based on the data distribution mode
    if data_distribution_mode == DATA_DISTRIBUTION_STATIC_MODE:
        # Static requests generation
        requests, timestamps_hours = generate_static_requests(config)
    else:
        # Dynamic requests generation
        requests, timestamps_hours = generate_dynamic_requests(config)

    # Create a dataframe where each row is composed of
    # a timestamp and the corresponding request
    df = create_dataframe(
        {
            TIMESTAMP_COLUMN: timestamps_hours[: len(requests)],
            REQUEST_COLUMN: requests,
        }
    )

    # Save just created dataframe
    # as dataset
    save_dataset(df, DATASET_RAW_TYPE, config)

    # Prepare save paths
    zipf_log_log_save_path = (
            PLOTS_SAVE_PATH
            / data_distribution_mode
            / ZIPF_LOGLOG_SAVE_PATH_NAME
    )
    daily_profile_save_path = (
            PLOTS_SAVE_PATH
            / data_distribution_mode
            / DAILY_PROFILE_SAVE_PATH_NAME
    )
    key_usage_heatmap_save_path = (
            PLOTS_SAVE_PATH / data_distribution_mode / KEY_USAGE_SAVE_PATH_NAME
    )

    # Show data generation -related plots
    plot_zipf_loglog(requests, zipf_log_log_save_path)
    plot_daily_profile(timestamps_hours, daily_profile_save_path)
    plot_key_usage_heatmap(
        min_key, max_key, requests, timestamps_hours, key_usage_heatmap_save_path
    )

    info("Data generation completed")
