"""main.py

Main entry point and orchestrator for the entire pipeline.

This module defines the high-level sequence of steps for data handling,
model training, validation, testing, and simulation, by calling the
respective functions from the pipeline steps.

Functions:
    main() -> None
        Executes the full pipeline workflow in sequence: data generation,
        preprocessing, validation, training, testing, and simulations.
"""

from pipeline.steps.data_generator import generate_data
from pipeline.steps.data_preprocessor import preprocess_data
from pipeline.steps.simulator import run_simulations
from pipeline.steps.tester import test_model
from pipeline.steps.trainer import train_model
from pipeline.steps.validator import validate_model


def main() -> None:
    """Run the full pipeline.

    This function orchestrates the entire pipeline,
    running the following steps in sequence:
    1. Data generation.
    2. Data preprocessing.
    3. Model validation.
    4. Model training.
    5. Model testing.
    6. Simulations.

    Returns:
        None
    """
    # -----------------------
    # Pipeline
    # -----------------------

    # (1) Data generation
    generate_data()

    # (2) Data preprocessing
    preprocess_data()

    # (3) Validation
    validate_model()

    # (4) Training
    train_model()

    # (5) Testing
    test_model()

    # (6) Simulations
    run_simulations()


if __name__ == "__main__":
    main()
