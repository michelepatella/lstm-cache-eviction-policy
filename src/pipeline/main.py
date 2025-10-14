from pipeline.steps.data_generation.data_generator import generate_data
from pipeline.steps.data_preprocessing.data_preprocessor import preprocess_data
from pipeline.steps.simulation.simulator import run_simulations
from pipeline.steps.testing.tester import test_model
from pipeline.steps.training.trainer import train_model
from pipeline.steps.validation.validator import validate_model
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

    # (3) Model validation
    validate_model()

    # (4) Model training
    train_model()

    # (5) Model testing
    test_model()

    # (6) Simulation
    run_simulations()


if __name__ == "__main__":
    main()
