from simulation.caches.utils.last_accesses_extractor import get_last_accesses
from utils.logs.levels.info_logger import info
from utils.simulation.classes.BaseCache import BaseCache


class LSTMCache(BaseCache):
    def __init__(
        self,
        cache_class,
        metrics_logger,
        config_settings,
    ):
        super().__init__(
            cache_class,
            metrics_logger,
            config_settings,
        )

        self.threshold_score = config_settings.simulation.lstm.threshold

        info("LSTM cache initialized.")

    def evict_key(self, key):
        self.store.pop(key, None)
        self.expiry.pop(key, None)
        self.scores.pop(key, None)

    def _put_key(
        self,
        key,
        score,
        current_time,
    ):
        self.store[key] = key
        self.scores[key] = score
        self.expiry[key] = current_time + self.ttl
        self.metrics_logger.log_put(key, current_time, self.ttl)

    def put(
        self,
        key,
        score,
        current_time
    ):
        self._remove_expired_keys(current_time)

        if key not in self.store and len(self.store) >= self.maxsize:
            # Get the sequence length to be extracted
            seq_len = config.model.sequence.length

            # Extract seed sequence
            seed_seq = get_last_accesses(
                current_idx,
                seq_len
                testing_set,
                config_settings,
            )

        self._put_key(key, score, current_time)

        # print a successful message
        info("🟢 Key inserted.")
