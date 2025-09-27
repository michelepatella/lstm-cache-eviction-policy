from lstm_eviction_policy.config import (
    prepare_config,
)
from lstm_eviction_policy.utils.logs.logs_setup import setup_logs

# 0. LOGS SETUP
setup_logs()

# 1. PREPARE CONFIGURATION SETTINGS
config = prepare_config()

# 2. GENERATE SYNTHETIC DATA
#generate_data(config)

# 3. PREPROCESS GENERATED DATA
#preprocess_data(config)

# 4. FIND THE BEST HYPERPARAMETERS
#config = validate_model(config)

# 5. TRAIN THE MODEL
# training(config)

# 6. TEST THE STANDALONE MODEL
# testing(config)

# 7. COMPARE THE FRAMEWORK AGAINST BASELINE CACHES
# run_simulations(config)
