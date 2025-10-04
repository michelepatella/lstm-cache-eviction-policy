from pipeline.config import prepare_config
from simulation import run_simulations
from utils.logs.initializer import initialize_logs


def main():
    # 0. LOGS SETUP
    initialize_logs()

    # 1. PREPARE CONFIGURATION SETTINGS
    config = prepare_config()

    # 2. GENERATE SYNTHETIC DATA
    # generate_data(config)

    # 3. PREPROCESS GENERATED DATA
    # preprocess_data(config)

    # 4. FIND THE BEST HYPERPARAMETERS
    # config = validate_model(config)

    # 5. TRAIN THE MODEL
    # train_model(config)

    # 6. TEST THE STANDALONE MODEL
    # test_model(config)

    run_simulations(config)


if __name__ == "__main__":
    main()
