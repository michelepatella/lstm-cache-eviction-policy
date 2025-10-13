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

    # -----------------------
    # Pipeline
    # -----------------------
    # (1) Data generation
    generate_data()

    # (2) Data preprocessing
    preprocess_data()
    return
    # (3) Model validation
    validate_model()

    # (4) Model training
    train_model()

    # (5) Model testing
    test_model()

    # -----------------------
    # Simulation
    # -----------------------
    run_simulations()


if __name__ == "__main__":
    main()
