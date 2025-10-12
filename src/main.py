from config import prepare_config
from pipeline.data_generation.data_generator import generate_data
from pipeline.data_preprocessing.data_preprocessor import preprocess_data
from pipeline.testing.tester import test_model
from pipeline.training.trainer import train_model
from simulation import run_simulations
from utils.logs.initializer import initialize_logs


def main():
    initialize_logs()

    config = prepare_config()

    generate_data(config)

    preprocess_data(config)

    #config = validate_model(config)

    train_model(config)

    test_model(config)

    #run_simulations(config)


if __name__ == "__main__":
    main()
