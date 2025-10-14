from pipeline.const import (
    DAILY_PROFILE_PLOT_FILE_NAME,
    DATA_DISTRIBUTION_STATIC_MODE,
    DATASET_RAW_TYPE,
    KEY_USAGE_HEATMAP_FILE_NAME,
    LOGS_DATA_GENERATION_PHASE,
    PLOTS_DIRECTORY_PATH,
    REQUEST_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
    ZIPF_LOG_LOG_PLOT_FILE_NAME,
)
from pipeline.config.configurator import prepare_config
from pipeline.steps.data_generation.requests.dynamic_generator import (
    generate_dynamic_requests,
)
from pipeline.steps.data_generation.requests.static_generator import (
    generate_static_requests,
)
from utils.dataset.builder import create_dataset
from pipeline.steps.data_generation.visualization.plots.daily_profile_plotter import (
    plot_daily_profile,
)
from pipeline.steps.data_generation.visualization.plots.key_usage_heatmap_plotter import (
    plot_key_usage_heatmap,
)
from pipeline.steps.data_generation.visualization.plots.zipf_loglog_plotter import (
    plot_zipf_loglog,
)
from utils.dataset.locator import get_dataset_abs_path
from utils.dataset.saver import save_dataset
from utils.logs.initializer import logs_phase
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


def generate_data() -> None:
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

    Returns:
        None
    """
    # Set the new state
    logs_phase.set(LOGS_DATA_GENERATION_PHASE)

    # Read configuration
    config = prepare_config()

    # Prepare configuration
    data_distribution_mode = config.data.generation.mode
    min_key = config.data.generation.keys.min
    max_key = config.data.generation.keys.max

    debug(f"Data distribution mode: {data_distribution_mode}")

    # Generate requests with corresponding timestamps,
    # based on the data distribution mode
    if data_distribution_mode == DATA_DISTRIBUTION_STATIC_MODE:
        # Static requests generation
        requests, timestamps_hours = generate_static_requests(config)
    else:
        # Dynamic requests generation
        requests, timestamps_hours = generate_dynamic_requests(config)

    # Create a dataset where each row is composed of
    # a timestamp and the corresponding request
    df = create_dataset(
        {
            TIMESTAMP_COLUMN_NAME: timestamps_hours[: len(requests)],
            REQUEST_COLUMN_NAME: requests,
        }
    )

    # Retrieve path where
    # to save dataset
    dataset_path = get_dataset_abs_path(DATASET_RAW_TYPE, data_distribution_mode)

    # Save just created dataset
    save_dataset(df, dataset_path)

    # Prepare save paths
    zipf_log_log_save_path = (
        PLOTS_DIRECTORY_PATH
        / data_distribution_mode
        / ZIPF_LOG_LOG_PLOT_FILE_NAME
    )
    daily_profile_save_path = (
        PLOTS_DIRECTORY_PATH
        / data_distribution_mode
        / DAILY_PROFILE_PLOT_FILE_NAME
    )
    key_usage_heatmap_save_path = (
        PLOTS_DIRECTORY_PATH
        / data_distribution_mode
        / KEY_USAGE_HEATMAP_FILE_NAME
    )

    # Show data generation -related plots
    plot_zipf_loglog(requests, zipf_log_log_save_path)
    plot_daily_profile(timestamps_hours, daily_profile_save_path)
    plot_key_usage_heatmap(
        min_key,
        max_key,
        requests,
        timestamps_hours,
        key_usage_heatmap_save_path,
    )

    info("Data generation completed")


if __name__ == "__main__":
    generate_data()
