from lstm_eviction_policy.config import (
    prepare_config,
)
from lstm_eviction_policy.data_generation import (
    generate_data,
)
from lstm_eviction_policy.data_preprocessing import (
    preprocess_data,
)

# PIPELINE
# 0. PREPARE CONFIGURATION SETTINGS
config = prepare_config()

# 1. GENERATE SYNTHETIC DATA
generate_data(config)

# 2. PREPROCESS GENERATED DATA
preprocess_data(config)

# 3. FIND THE BEST HYPERPARAMETERS
# config = validation(config)

# 4. TRAIN THE MODEL
# training(config)

# 5. TEST THE STANDALONE MODEL
# testing(config)

# 6. COMPARE THE FRAMEWORK AGAINST BASELINE CACHES
# run_simulations(config)
