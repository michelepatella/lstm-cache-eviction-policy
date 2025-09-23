import numpy as np
from matplotlib import pyplot as plt


def plot_daily_profile(timestamps, bin_size=0.5):
    try:
        # extract hours
        hours = np.array(timestamps) % 24

        # define bins
        num_bins = int(24 / bin_size)
        bins = np.linspace(0, 24, num_bins + 1)

        # define the histogram
        counts, _ = np.histogram(hours, bins=bins)

        plt.figure(figsize=(12, 10))
        plt.bar(bins[:-1], counts, width=bin_size, align="edge", edgecolor="black")
        plt.xlabel("Hour of Day", fontsize=16)
        plt.ylabel("Number of Requests", fontsize=16)
        plt.title("Distribution of Requests Over 24 Hours", fontsize=18)
        plt.xticks(np.arange(0, 25, step=1), fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.close()

    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")


def plot_key_usage_heatmap(requests, timestamps, config):
    try:
        # define the heatmap
        heatmap = np.zeros((24, config.data.general.keys.max - 1), dtype=int)

        # fill the heatmap
        for key, ts in zip(requests, timestamps):
            hour = int(ts)
            key_idx = key - config.data.general.keys.min
            if 0 <= hour < 24 and 0 <= key_idx < config.data.general.keys.max - 1:
                heatmap[hour, key_idx] += 1

        plt.figure(figsize=(12, 10))
        plt.imshow(heatmap, aspect="auto", cmap="viridis")
        plt.colorbar(label="Access Count")
        plt.xlabel("Key", fontsize=16)
        plt.ylabel("Hour of Day", fontsize=16)
        plt.title("Heatmap of Key Access Frequency by Hour of Day", fontsize=18)
        plt.yticks(
            ticks=np.arange(24), labels=[f"{h}:00" for h in range(24)], fontsize=16
        )
        plt.xticks(
            ticks=np.arange(
                0,
                config.data.general.keys.max - 1,
                max(1, config.data.general.keys.max - 1 // 20),
            ),
            labels=np.arange(
                config.data.general.keys.min,
                config.data.general.keys.max,
                max(1, config.data.general.keys.max - 1 // 20),
            ),
            rotation=90,
            fontsize=16,
        )
        plt.tight_layout()
        plt.show()
        plt.close()
    except AttributeError as e:
        raise AttributeError(f"AttributeError: {e}.")
    except TypeError as e:
        raise TypeError(f"TypeError: {e}.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}.")
    except IndexError as e:
        raise IndexError(f"IndexError: {e}.")
    except Exception as e:
        raise RuntimeError(f"RuntimeError: {e}.")
