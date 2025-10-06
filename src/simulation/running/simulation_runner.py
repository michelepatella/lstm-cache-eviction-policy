from tqdm import tqdm

from const import HIT_COUNTER_NAME
from lstm_cache_eviction_policy.management.lstm_manager import (
    manage_lstm_cache,
)
from utils.simulation.key_in_cache_searcher import search_key_in_cache
from simulation.metrics.calculator import calculate_cache_simulation_metrics
from simulation.initialization.initializer import initialize_simulation
from simulation.running.simulation_tracer import trace_hits_misses
from simulation.utils.time_key_from_row_extractor import preprocess_row


def run_cache_simulation(
    cache,
    policy,
    metrics_logger,
    config,
):
    # Setup for simulation
    (
        counters,
        timeline,
        recent_hits,
        cache_latencies,
        testing_set,
        testing_loader,
    ) = initialize_simulation(config)

    # Iterate over testing set, assuming each
    # row represents a request to be satisfied
    for idx in tqdm(
        range(len(testing_set)),
        desc=f"Simulating {policy}",
    ):
        # Extract the current row
        # from the dataset
        row = testing_set[idx]

        # Get the number of hits so far
        prev_hits_count = counters[HIT_COUNTER_NAME]

        # Extrapolate current time and requested
        # key from the current row
        current_time, key = preprocess_row(row)

        # if the LSTM cache is being used
        if policy == "LSTM":
            (
                cache_latency,
                num_prefetch,
            ) = manage_lstm_cache(
                cache,
                key,
                current_time,
                idx,
                counters,
                device,
                model,
                testing_set,
                config,
            )

            # store cache latency
            cache_latencies.append(cache_latency)

        # if the traditional cache (LRU, LFU, FIFO, or RANDOM) is being used
        else:
            # search the key into the cache
            is_hit = search_key_in_cache(cache, key, current_time, counters)

            if not is_hit:
                # put a key in cache
                cache.put(key, current_time)

        # update number of hits and misses
        (recent_hits, timeline) = trace_hits_misses(
            counters,
            prev_hits_count,
            recent_hits,
            window,
            idx,
            timeline,
        )

    # compute cache metrics
    (
        hit_rate,
        miss_rate,
        eviction_mistake_rate,
        avg_cache_latency,
    ) = calculate_cache_simulation_metrics(
        counters, cache_latencies, metrics_logger, config
    )

    return {
        "policy": policy,
        "hit_rate": hit_rate,
        "miss_rate": miss_rate,
        "hits": counters["hits"],
        "misses": counters["misses"],
        "avg_prefetching_latency": avg_cache_latency,
        "timeline": timeline,
        "eviction_mistake_rate": eviction_mistake_rate,
    }
