# LSTM Cache Eviction Policy

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)

![Git](https://img.shields.io/badge/Code_Versioning-Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![DVC](https://img.shields.io/badge/Data_Versioning-DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/Model_&_Experiments_Versioning-MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)

![Coverage](https://img.shields.io/badge/Coverage-72%25-green?style=for-the-badge&logo=codecov&logoColor=white)
![Pytest](https://img.shields.io/badge/Tests-PyTest-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)
![Great Expectations](https://img.shields.io/badge/Data_Tests-Great_Expectations-FF6433?style=for-the-badge&logo=data-robot&logoColor=white)
![Deepchecks](https://img.shields.io/badge/Data_&_Model_Tests-Deepchecks-00E5B9?style=for-the-badge&logoColor=white)

![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)

![Docker](https://img.shields.io/badge/Containerization-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Orchestration-K8s-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Deployment-Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

![Better Uptime](https://img.shields.io/badge/Status-Better_Uptime-232F3E?style=for-the-badge&logoColor=white)
![Locust](https://img.shields.io/badge/Load_Testing-Locust-76923C?style=for-the-badge&logoColor=white)
![Prometheus](https://img.shields.io/badge/Metrics-Prometheus-e6522c?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Dashboards-Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white)

This [project](https://github.com/michelepatella/lstm-cache-eviction-policy) applies **MLOps** practices to a Sequence Prediction task, consisting of predicting the next access data likelihood in a sequence through a LSTM-based model, producing survival and uncertainty-aware data scores used for guiding **cache eviction decisions**.

By serving this as a **microservice-based API**, the system replaces rigid, rule-based policies (like LRU or LFU) that suffer from inefficient resource usage and high costs due to their lack of "future" insight. Furthermore, it addresses a critical gap in current SOTA Deep Learning solutions by incorporating model uncertainty, ensuring that eviction decisions are not only data-driven but statistically reliable.

---

## MLOps
The project focuses on the application of the MLOps practices across the following, topic-based, project milestones:


### Inception

- Overview of the ML system's requirements, covering data, performance, and quality requirements.
- Self-assessment of our ML system under the EU [AI Act](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai), identifying its risk category and applicable regulatory decisions.
- Specification of our system through [ML Canvas](docs/ml_canvas.pdf) by Louis Dorard.
- Overview of the project structure, following a Cookiecutter-like layout, standard convention, and best practices.

> More details available [here](https://michelepatella.gitbook.io/lstm-cache-eviction-policy-documentation/D5QOUAgv73B41REdHZx1/inception).

### Reproducibility

- Tracking and controlling changes in the code using [Git](https://github.com), following well-defined strategies, and best practices.
- Version and track data, models, and experiments with [DVC](https://dagshub.com/michelepatella/lstm-cache-eviction-policy?filter=dvc) and [MLflow](https://dagshub.com/michelepatella/lstm-cache-eviction-policy.mlflow/), integrated via [DagsHub](https://dagshub.com/michelepatella/lstm-cache-eviction-policy/), synchronized with the [GitHub repository](https://github.com/michelepatella/lstm-cache-eviction-policy).
  - How do we use [DVC](https://dagshub.com/michelepatella/lstm-cache-eviction-policy?filter=dvc) with [DagsHub](https://dagshub.com/michelepatella/lstm-cache-eviction-policy/) for tracking our [data](data) and [models](models) produced by an orchestrated, [declarative pipeline](dvc.yaml).
  - How do we use [MLflow Tracking](https://dagshub.com/michelepatella/lstm-cache-eviction-policy.mlflow/#/experiments) and [MLflow Model Registry](https://dagshub.com/michelepatella/lstm-cache-eviction-policy.mlflow/#/models) for tracking [our experiments](https://dagshub.com/michelepatella/lstm-cache-eviction-policy.mlflow/#/experiments/0/runs?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D) and analyzing results, and storing in a centralized way our ML models, respectively.
> More details available [here](https://michelepatella.gitbook.io/lstm-cache-eviction-policy-documentation/D5QOUAgv73B41REdHZx1/reproducibility).

### Quality Assurance

- Introduction to the static analysis tools and their application in the project workflow, ensuring integrity, quality, consistency, and security.
- Overview of dynamic analysis conducted in this project.
  - Overview of our [offline testing](tests) approach, comprising [code](tests/code), [data](tests/data), [model](tests/model), and [API testing](tests/api), while following the Arrange-Act-Assert framework.
    - Description of the [code testing](tests/code) strategy adopted.
    - Description of [data testing](tests/data) types designed and implemented, covering [data integrity](tests/data/integrity), [quality](tests/data/quality), and [splitting](tests/data/train_test_validation).
    - Description of our [model testing](tests/model) strategy, involving [training](tests/model/training), [inference](tests/model/inference), [performance](tests/model/performance), and [behavioral tests](tests/model/behavioral).
    - Description of [API testing](tests/api) strategy adopted for ensuring robustness, artifact integrity, and system acceptance.
  - Overview of the adopted [online testing](src/api/services/predictor/predictor_service.py) strategy, consisting of canary testing, and the corresponding feedback loop.
> More details available [here](https://michelepatella.gitbook.io/lstm-cache-eviction-policy-documentation/D5QOUAgv73B41REdHZx1/quality-assurance).

### API

- Description of the ML-based [API](src/api) architecture, its components, and their interactions.
- Breakdown of the specialized [microservices](src/api/services) that power the ML eviction pipeline.
  - Description of the [API gateway](src/api/gateway): the central orchestrator of the whole API pipeline.
  - Presentation of the [data preprocessing layer](src/api/services/featurizer) responsible for transforming raw data into model-ready tensors.
  - Overview of the [core inference engine](src/api/services/predictor) responsible for executing confidence-aware autoregressive rollouts.
  - Introduction to the [decision-support logic](src/api/services/scorer) responsible for calculating item importance scores based on predicted survival and model confidence.
  - Overview of the [final execution layer](src/api/services/decider) responsible for selecting eviction candidates based on importance scores and system constraints.
- Presentation of [our web API](http://34.147.98.233:80) via the standard, machine-readable specification language OpenAPI.
- Introduction to the standardized documentation providing transparency on the [data](data) and [models](models) powering the inference pipeline.
  - [Model Card](models/README.md) for LSTM-Based Data Access Predictor.
  - [Dataset Card](data/README.md) for Synthetic Data Accesses.

> More details available [here](https://michelepatella.gitbook.io/lstm-cache-eviction-policy-documentation/D5QOUAgv73B41REdHZx1/api).

### Deployment

- Description of how to transform standalone application services into a production-ready ecosystem, leveraging Docker and Docker Compose for containerization and [orchestration](docker-compose.yml), respectively.
- Presentation of the microservices orchestration into a production cluster, using declarative [Kubernetes manifests](k8s).
- Description of the cloud-based deployment infrastructure—Google Cloud—and its configuration workflow.
- Introduction to the CI/CD pipelines with [GitHub Actions](https://github.com/michelepatella/lstm-cache-eviction-policy/actions) for automating tasks ranging from QA and building activities to deployment.
  - Description of the [CI pipeline](.github/workflows/ci.yml) to automate static analysis, security scans, testing, and build.
  - Presentation of the [CD pipeline](.github/workflows/cd.yml) to manage reliable deployment of the API and [documentation](docs).

> More details available [here](https://michelepatella.gitbook.io/lstm-cache-eviction-policy-documentation/D5QOUAgv73B41REdHZx1/deployment).

### Monitoring

- Introduction to API resource-level monitoring with [Better Uptime](https://lstmcachevictionpolicy.betteruptime.com/), [Locust](load_test), [Prometheus, and Grafana](prometheus-grafana ).
  - Discussion of global availability monitoring and incident management with multi-region health checks and real-time alerting via [Better Uptime](https://lstmcachevictionpolicy.betteruptime.com/).
  - Introduction to [performance and stress tests](load_test) to evaluate API response times, throughput, and system stability under concurrent user traffic.
  - Description of the [metrics collection activity](src/api/gateway/main.py) and corresponding visualization through [dashboards](prometheus-grafana/dashboards) via [Prometheus and Grafana](prometheus-grafana), respectively.
- Introduction to system performance-level monitoring with Alibi Detect and Deepchecks.
  - Overview of our [drift detection strategy](.github/workflows/drift_detection.yml), based on the Alert-Inspect-Act framework.
    - How we detect [univariate drift](src/components/model/monitoring/drift/variants/univariate_drift_detector.py) in our system with Alibi Detect.
    - How we detect [multivariate drift](src/components/model/monitoring/drift/variants/multivariate_drift_detector.py) in our system with Alibi Detect.
    - How we detect [target drift](src/components/model/monitoring/drift/variants/target_drift_detector.py) in our system with Alibi Detect.
    - How we detect [prediction drift](src/components/model/monitoring/drift/variants/prediction_drift_detector.py) in our system with Alibi Detect.
  - Overview of our [model performance monitoring](.github/workflows/model_performance_monitoring.yml) strategy, based on the Alert-Inspect-Act framework.

> More details available [here](https://michelepatella.gitbook.io/lstm-cache-eviction-policy-documentation/D5QOUAgv73B41REdHZx1/monitoring).

<br>

#### The full documentation is available [here](https://michelepatella.gitbook.io/lstm-cache-eviction-policy-documentation/D5QOUAgv73B41REdHZx1).

---
## Author

[![Michele Patella](https://img.shields.io/badge/Author-Michele%20Patella-orange?style=for-the-badge&logo=github&logoColor=white)](https://github.com/michelepatella)
