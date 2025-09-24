from const import (
    DATA_DISTRIBUTION_STATIC_MODE,
    PIPELINE_PHASE_DATA_GENERATION,
    REQUEST_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
)
from lstm_eviction_policy.config.classes.Config import (
    Config,
)
from lstm_eviction_policy.data_generation.generation.requests.dynamic_requests_generator import (
    generate_dynamic_requests,
)
from lstm_eviction_policy.data_generation.generation.requests.static_requests_generator import (
    generate_static_requests,
)
from lstm_eviction_policy.data_generation.utils.dataframe_builder import (
    create_dataframe,
)
from lstm_eviction_policy.data_generation.utils.dataset_saver import (
    save_dataset,
)
from lstm_eviction_policy.data_generation.visualization.daily_profile_plotter import (
    plot_daily_profile,
)
from lstm_eviction_policy.data_generation.visualization.key_usage_heatmap_plotter import (
    plot_key_usage_heatmap,
)
from lstm_eviction_policy.data_generation.visualization.zipf_loglog_plotter import (
    plot_zipf_loglog,
)
from lstm_eviction_policy.utils.logs.log_utils import (
    debug,
    info,
    phase_var,
)


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
    phase_var.set(PIPELINE_PHASE_DATA_GENERATION)

    # Get the data distribution
    # mode (static or dynamic)
    data_distribution_mode = config.data.general.mode

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
            TIMESTAMP_COLUMN_NAME: timestamps_hours[: len(requests)],
            REQUEST_COLUMN_NAME: requests,
        }
    )

    # Save just created dataframe
    # as dataset
    save_dataset(df, config)

    # Show data generation -related plots
    plot_zipf_loglog(requests)
    plot_daily_profile(timestamps_hours)
    plot_key_usage_heatmap(requests, timestamps_hours, config)

    info("Data generation completed")
