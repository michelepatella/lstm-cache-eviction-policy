from typing import Dict, Tuple, List

from torch.utils.data import DataLoader

from config.classes.Config import Config
from const import HIT_COUNTER_NAME, MISS_COUNTER_NAME, TESTING_SPLIT_TYPE
from utils.data_loader.initializer import initialize_data_loader
from utils.dataset.AccessLogsDataset import AccessLogsDataset
from utils.logs.levels.debug_logger import debug
from utils.logs.levels.info_logger import info


def initialize_simulation(
    config: Config,
) -> Tuple[
    Dict[str, int],
    List[float],
    List[float],
    AccessLogsDataset,
    DataLoader,
]:
    """
    Initialize simulation data and structures for cache evaluation.

    This function sets up all necessary structures and datasets for
    running a cache simulation. It initializes counters, timelines,
    cache latency records, and prepares the testing dataset and data loader.

    Parameters:
        config (Config): Configuration object.

    Returns:
        Tuple[Dict[str, int], List[float], List[float], AccessLogsDataset, DataLoader]:
            A tuple containing a dictionary with initialized cache hit
            and miss counters, a list to track cache events over time,
            a list to store latency of cache accesses, a dataset for
            testing, and a data loader for iterating over the testing set.
    """
    # Prepare configuration
    testing_batch_size = config.testing.batch_size
    testing_shuffle = config.testing.shuffle

    # Initialize data
    counters = {
        HIT_COUNTER_NAME: 0,
        MISS_COUNTER_NAME: 0,
    }
    timeline = []
    cache_latencies = []

    # Get testing set
    testing_set, testing_loader = initialize_data_loader(
        TESTING_SPLIT_TYPE,
        testing_batch_size,
        testing_shuffle,
        config,
        AccessLogsDataset,
    )

    debug(f"Testing size for simulation: {len(testing_set)}")

    info("Simulation initialization completed")

    return (
        counters,
        timeline,
        cache_latencies,
        testing_set,
        testing_loader,
    )
