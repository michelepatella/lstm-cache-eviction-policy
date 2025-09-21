from lstm_eviction_policy.config import prepare_config

# PIPELINE
# 0. PREPARE CONFIGURATION SETTINGS
config_settings = prepare_config()

# 1. GENERATE SYNTHETIC DATA
# data_generation(config_settings)

# 2. PREPROCESS GENERATED DATA
# data_preprocessing(config_settings)

# 3. FIND THE BEST HYPERPARAMETERS
# config_settings = validation(config_settings)

# 4. TRAIN THE MODEL
# training(config_settings)

# 5. TEST THE STANDALONE MODEL
# testing(config_settings)

# 6. COMPARE THE FRAMEWORK AGAINST BASELINE CACHES
# run_simulations(config_settings)
