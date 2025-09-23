from collections import Counter

import numpy as np
from matplotlib import pyplot as plt


def plot_zipf_loglog(requests: list[int]) -> None:
    # Count how many keys are
    # involved in generated requests
    key_counts = Counter(requests)

    # Get key access frequencies
    key_frequencies = np.array(sorted(key_counts.values(), reverse=True))

    # Define ranks ranging from
    # the first place to the last one
    ranks = np.arange(1, len(key_frequencies) + 1)

    # Plot the Zipfian distribution
    # log-log plot
    plt.figure()
    plt.loglog(ranks, key_frequencies, marker="o")
    plt.title("Zipf Distribution (Log-Log)", fontsize=18)
    plt.xlabel("Key", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.close()
