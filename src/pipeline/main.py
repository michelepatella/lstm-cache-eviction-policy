import mlflow

from pipeline.config.configurator import prepare_config
from pipeline.steps.data_generator import generate_data
from pipeline.steps.data_preprocessor import preprocess_data
from pipeline.steps.simulator import run_simulations
from pipeline.steps.tester import test_model
from pipeline.steps.trainer import train_model
from pipeline.steps.validator import validate_model


def main():
    # -----------------------
    # Pipeline
    # -----------------------
    mlflow.start_run()

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

    # (6) Simulations
    run_simulations()

    mlflow.log_params(prepare_config().model_dump())
    mlflow.end_run()


if __name__ == "__main__":
    main()
