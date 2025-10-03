from utils.data.AccessLogsDataset import AccessLogsDataset
from utils.logs.levels.info_logger import info
from utils.model.initialization.trained_model_initializer import initialize_trained_model
from utils.data.data_loader.initializer import initialize_data_loader


def simulation_setup(policy_name, config_settings):
    """
    Method to set up the simulation environment.
    :param policy_name: The policy name.
    :param config_settings: The configuration settings.
    :return: All the data needed to run the simulation.
    """
    # initial message
    info("🔄 Simulation setup started...")

    # initialize data
    (device, criterion, model) = None, None, None
    counters = {
        "hits": 0,
        "misses": 0,
        "hits_cold_start": 0,
    }
    timeline = []
    recent_hits = []
    prefetching_latency = []
    window = config_settings.simulation.lstm.prediction.interval

    # get the testing set
    (testing_set, testing_loader) = initialize_data_loader(
        "testing",
        config_settings.testing.batch_size,
        False,
        config_settings,
        AccessLogsDataset,
    )

    # initial model setup, in case of LSTM cache
    if policy_name == "LSTM":
        # setup for lstm cache
        (device, criterion, model) = initialize_trained_model(
            testing_loader, config_settings
        )

        try:
            model.eval()
            model.to(device)
        except AttributeError as e:
            raise AttributeError(f"AttributeError: {e}.")
        except NameError as e:
            raise NameError(f"NameError: {e}.")
        except TypeError as e:
            raise TypeError(f"TypeError: {e}.")
        except Exception as e:
            raise RuntimeError(f"RuntimeError: {e}.")

    # print a successful message
    info("🟢 Simulation setup completed.")

    return (
        counters,
        timeline,
        recent_hits,
        prefetching_latency,
        window,
        testing_set,
        testing_loader,
        device,
        criterion,
        model,
    )
