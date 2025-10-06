from config import prepare_config
from pipeline.data_generation.data_generator import generate_data
from pipeline.data_preprocessing.data_preprocessor import preprocess_data
from pipeline.testing.tester import test_model
from pipeline.training.trainer import train_model
from simulation import run_simulations
from utils.logs.initializer import initialize_logs


def main():
    # 0. LOGS SETUP
    initialize_logs()

    # 1. PREPARE CONFIGURATION SETTINGS
    config = prepare_config()

    # 2. GENERATE SYNTHETIC DATA
    #generate_data(config)

    # 3. PREPROCESS GENERATED DATA
    #preprocess_data(config)

    # 4. FIND THE BEST HYPERPARAMETERS
    # config = validate_model(config)

    # 5. TRAIN THE MODEL
    train_model(config)

    # 6. TEST THE STANDALONE MODEL
    test_model(config)

    #run_simulations(config)


if __name__ == "__main__":
    main()
