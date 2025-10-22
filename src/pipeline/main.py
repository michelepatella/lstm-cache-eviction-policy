import dagshub

from src.const import (
    DAGS_HUB_MLFLOW_ENABLED,
    DAGS_HUB_REPO_NAME,
    DAGS_HUB_REPO_OWNER,
    MLFLOW_MAIN_RUN_NAME,
)
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

    dagshub.init(
        repo_owner=DAGS_HUB_REPO_OWNER,
        repo_name=DAGS_HUB_REPO_NAME,
        mlflow=DAGS_HUB_MLFLOW_ENABLED,
    )

    import mlflow
    with mlflow.start_run(run_name=MLFLOW_MAIN_RUN_NAME):

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

        # Experiment tracking
        mlflow.log_params(prepare_config().model_dump())
        mlflow.end_run()


if __name__ == "__main__":
    main()
