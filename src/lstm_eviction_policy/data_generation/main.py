from const import (
    DATA_DISTRIBUTION_STATIC_MODE,
    PIPELINE_PHASE_DATA_GENERATION,
    REQUEST_COLUMN_NAME,
    TIMESTAMP_COLUMN_NAME,
)
from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.data_generation.generation.requests.dynamic_requests_generator import (
    generate_dynamic_requests,
)
from lstm_eviction_policy.data_generation.generation.requests.static_requests_generator import (
    generate_static_requests,
)
from lstm_eviction_policy.data_generation.utils.dataframe_builder import (
    create_dataframe,
)
from lstm_eviction_policy.data_generation.utils.dataset_saver import save_dataset
from lstm_eviction_policy.data_generation.visualization.data_generation_plotter import (
    plot_daily_profile,
    plot_key_usage_heatmap,
)
from lstm_eviction_policy.data_generation.visualization.zipf_loglog_plotter import (
    plot_zipf_loglog,
)
from lstm_eviction_policy.utils.logs.log_utils import debug, info, phase_var


def data_generation(config: Config) -> None:
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
        requests, timestamps = generate_static_requests(config)

        # Set the path where to save
        # the dataset later
        dataset_path = config.data.dataset.paths.static
    else:
        # Dynamic requests generation
        requests, timestamps = generate_dynamic_requests(config)

        # Set the path where to save
        # the dataset later
        dataset_path = config.data.dataset.paths.dynamic

    # Create a dataframe where each row is composed of
    # a timestamp and the corresponding request
    df = create_dataframe(
        {
            TIMESTAMP_COLUMN_NAME: timestamps[: len(requests)],
            REQUEST_COLUMN_NAME: requests,
        }
    )

    # Save just created dataframe
    # as dataset
    save_dataset(df, dataset_path)

    # Show data generation -related plots
    plot_zipf_loglog(requests)
    plot_daily_profile(timestamps)
    plot_key_usage_heatmap(requests, timestamps, config)

    info("Data generation completed")
