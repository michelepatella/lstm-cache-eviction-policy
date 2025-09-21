from lstm_eviction_policy.config.classes.Config import Config
from lstm_eviction_policy.utils.logs.log_utils import debug, phase_var


def data_generation(config: Config) -> None:
    # Set the new pipeline state
    phase_var.set("data_generation")

    # Get the data distribution mode (static or dynamic)
    data_distribution_mode = config.data.general.mode

    debug(f"Data distribution mode: {data_distribution_mode}")

    # Generate requests with corresponding timestamps,
    # based on the data distribution mode
    if data_distribution_mode == "static":
        # Static requests generation
        requests, timestamps = generate_static_requests(config)
    else:
        # Dynamic requests generation
        requests, timestamps = generate_dynamic_requests(config)

    try:
        # Create a dataframe where each row is composed of
        # a timestamp and the corresponding request
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
    save_dataset(df, config)

    # show some plots
    plot_zipf_loglog(requests)
    plot_daily_profile(timestamps)
    plot_key_usage_heatmap(requests, timestamps, config)

    # show a successful message
    info("✅ Data generation successfully completed.")
