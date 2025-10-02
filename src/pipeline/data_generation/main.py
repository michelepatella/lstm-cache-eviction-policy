from const import (
    DATA_DISTRIBUTION_STATIC_MODE,
    LOGS_DATA_GENERATION_PHASE,
    REQUEST_COLUMN,
    TIMESTAMP_COLUMN,
    DATASET_RAW_TYPE,
)
from pipeline.config.classes.Config import (
    Config,
)
from pipeline.data_generation.generation.requests.dynamic_requests_generator import (
    generate_dynamic_requests,
)
from pipeline.data_generation.generation.requests.static_requests_generator import (
    generate_static_requests,
)
from pipeline.data_generation.utils.dataframe_builder import (
    create_dataframe,
)
from pipeline.data_generation.utils.dataset_saver import (
    save_dataset,
)
from pipeline.data_generation.visualization.daily_profile_plotter import (
    plot_daily_profile,
)
from pipeline.data_generation.visualization.key_usage_heatmap_plotter import (
    plot_key_usage_heatmap,
)
from pipeline.data_generation.visualization.zipf_loglog_plotter import (
    plot_zipf_loglog,
)
from pipeline.utils.logs.levels.debug_logger import debug
from pipeline.utils.logs.levels.info_logger import info
from pipeline.utils.logs.logs_setup import logs_phase


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
    # Set the new pipeline state
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

    # Show data generation -related plots
    plot_zipf_loglog(requests)
    plot_daily_profile(timestamps_hours)
    plot_key_usage_heatmap(min_key, max_key, requests, timestamps_hours)

    info("Data generation completed")
