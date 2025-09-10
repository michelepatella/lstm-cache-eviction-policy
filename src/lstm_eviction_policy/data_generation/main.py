from data_generation.generation.requests.dynamic_requests_generator import \
    generate_dynamic_requests
from data_generation.generation.requests.static_requests_generator import \
    generate_static_requests
from data_generation.utils.df_builder import create_dataframe
from data_generation.visualization.data_generation_plotter import (
    plot_daily_profile, plot_key_usage_heatmap, plot_zipf_loglog)
from utils.data.dataset.dataset_saver import save_dataset
from utils.logs.log_utils import debug, info, phase_var


def data_generation(config_settings):
    """
    Method to orchestrate data generation.
    :param config_settings: The configuration settings.
    :return:
    """
    # initial message
    info("🔄 Data generation started...")

    # set the variable indicating the state of the process
    phase_var.set("data_generation")

    # debugging
    debug(f"⚙️Type of distribution: {config_settings.distribution_type}.")

    if config_settings.distribution_type == "static":
        # generate static requests and timestamps
        requests, timestamps = generate_static_requests(config_settings)
    else:
        # generate dynamic requests and timestamps
        requests, timestamps = generate_dynamic_requests(config_settings)

    try:
        # create dataframe
        df = create_dataframe(
            {
                "timestamp": timestamps[: len(requests)],
                "request": requests,
            }
        )
    except NameError as e:
        raise NameError(f"NameError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except IndexError as e:
        raise IndexError(f"IndexError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")

    # save the dataset
    save_dataset(df, config_settings)

    # show some plots
    plot_zipf_loglog(requests)
    plot_daily_profile(timestamps)
    plot_key_usage_heatmap(requests, timestamps, config_settings)

    # show a successful message
    info("✅ Data generation successfully completed.")
