from config import prepare_config
from pipeline.data_generation.data_generator import generate_data
from pipeline.data_preprocessing.data_preprocessor import preprocess_data
from pipeline.testing.tester import test_model
from pipeline.training.trainer import train_model
from pipeline.validation.validator import validate_model
from simulation import run_simulations
from utils.logs.initializer import initialize_logs


def main():
    # -----------------------
    # Setup
    # -----------------------
    initialize_logs()
    config = prepare_config()

    # -----------------------
    # Pipeline
    # -----------------------
    # (1) Data generation
    generate_data(config)

    # (2) Data preprocessing
    preprocess_data(config)
    return
    # (3) Model validation
    config = validate_model(config)

    # (4) Model training
    train_model(config)

    # (5) Model testing
    test_model(config)

    # -----------------------
    # Simulation
    # -----------------------
    run_simulations(config)


if __name__ == "__main__":
    main()
