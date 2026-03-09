<!-- TOP -->
<a id="readme-top"></a>

<!-- HEADER -->
<div align="center">

  <!-- TITLE -->
  <h1 align="center">LSTM Cache Eviction Policy</h1>

  <!-- DESCRIPTION AND LINKS -->
  <p align="center">
    A microservice API implementing a production-grade cache eviction policy powered by deep learning.
    <br/>
    <br/>
    <a href="https://github.com/michelepatella/lstm-cache-eviction-policy">GitHub</a>
    &middot;
    <a href="https://dagshub.com/michelepatella/lstm-cache-eviction-policy">DagsHub</a>
    &middot;
    <a href="https://michelepatella.gitbook.io/lstm-cache-eviction-policy-documentation">Official Docs</a>
    &middot;
    <a href="https://michelepatella.github.io/lstm-cache-eviction-policy/">Code Reference</a>
  </p>

  <!-- BADGES -->
  ![CI](https://img.shields.io/github/actions/workflow/status/michelepatella/lstm-cache-eviction-policy/ci.yml?logo=github-actions&style=for-the-badge&label=CI&logoColor=white)
  ![Coverage](https://img.shields.io/badge/Coverage-72%25-yellow?style=for-the-badge&logo=pytest&logoColor=white)
</div>

<br/>

<!-- 'MOTIVATION & IMPACT' SECTION -->
## Motivation & Impact
**Limitations of Traditional Policies**  
Traditional cache eviction policies (e.g., LRU, LFU) rely only on past access patterns and cannot anticipate future demand.  
→ _Inefficient resource utilization_  
→ _Increased infrastructure and operational costs_  
→ _Degraded system performance and user experience_

**Limitations of Machine Learning-Based Policies**  
Machine learning-based policies provide predictive power but ignore model uncertainty.  
→ _Unreliable eviction decisions_  

**The Proposed Solution**  
This project introduces an uncertainty-aware, deep learning–driven cache eviction strategy, achieving:  
* **+14%** hit rate vs. LRU
* **+6%** hit rate vs. LFU
* **+1%** hit rate vs. uncertainty-unaware policy

<p align="right"><a href="#readme-top">Top ↑</a></p>

<!-- 'FEATURES' SECTION -->
## Features

<!-- 'CORE' SUBSECTION -->
### Core
- **Adaptive eviction** with autoregressive, multi-step LSTM capturing future data access patterns.
- **Uncertainty-aware scoring** protecting high-uncertainty data items using Monte Carlo dropout.
- **Modular architecture** separating featurization, prediction, scoring, and eviction logic.
- **Feature engineering** with cyclical time encoding and local recency/frequency for temporal locality.
- **Temporal-aware loss** prioritizing imminent-key predictions to minimize cache misses.
- **Efficient training** with hyperparameter search, early stopping, and parallel execution via Ray and PyTorch DDP.
- **Optimized inference** via 8-bit quantization and 20% weight pruning for faster CPU execution.

<!-- 'MLOPS' SUBSECTION -->
### MLOps

<!-- 'INCEPTION' SUBSECTION -->
#### Inception
- **ML requirements engineering** defining data, performance, and quality requirements.
- **AI governance self-assessment** to classify the system under the EU AI Act.
- **System specification** through ML Canvas to formalize objectives, stakeholders, and system constraints.

<!-- 'REPRODUCIBILITY' SUBSECTION -->
#### Reproducibility
- **Code versioning** with Git using a Git Flow-like branching strategy and pull requests-based integration.
- **Reproducible ML pipeline** defined with DVC for tracking pipeline stages.
- **Data and model versioning** with DVC to keep large artifacts outside Git.
- **Experiment tracking** with MLflow, logging parameters, metrics, and artifacts.
- **Model lifecycle management** through MLflow Model Registry, versioning and organizing production and staging models.

<!-- 'QUALITY ASSURANCE' SUBSECTION -->
#### Quality Assurance

- **Static analysis** enforcing code quality, formatting, and security via automated linters and vulnerability scanners.
- **Pre-commit checks** preventing secret leakage, enforcing repository hygiene, and standardizing formatting.
- **Comprehensive testing** of code, data, model, and API with unit, integration, system, and acceptance tests using pytest, Great Expectactions, and DeepChecks.
- **Data validation pipeline** ensuring dataset integrity and quality.
- **ML model testing suite** including training, inference, performance, and behavioral tests.
- **API reliability testing** validating robustness, artifact integrity, and full end-to-end prediction flows.
- **Production-safe deployment** with online canary testing and KPIs monitoring for models.

<!-- 'API' SUBSECTION -->
#### API

- **Deep learning microservice API** exposing RESTful endpoints and incapsulating the full ML pipeline using FastAPI.
- **API Gateway** orchestrating gRPC microservices with request validation (Pydantic), configuration, standardized responses, and monitoring.
- **Featurizer Service** transforming raw data into model-ready tensors with feature engineering.
- **Predictor Service** managing model versions and executing confidence-aware autoregression with online canary testing.
- **Scorer Service** computing hybrid survival- and uncertainty-aware scores for data items based on predictions and uncertainties.
- **Decider Service** applying operational constraints and selecting top-K eviction candidates based on scores.
- **Documented artifacts** with model and dataset cards, plus OpenAPI specification.

<!-- 'DEPLOYMENT' SUBSECTION -->
#### Deployment

- **Containerization** packaging each microservice into Docker containers with isolated dependencies.  
- **Docker Compose orchestration** defining and running all containers together.  
- **Kubernetes deployment** orchestrating containers in a production cluster with secure manifests.  
- **Cloud infrastructure** leveraging a managed Google Cloud Autopilot Kubernetes cluster and Artifact Registry for hosting.  
- **CI pipeline automation** building, testing, scanning, and publishing Docker images via GitHub Actions.  
- **CD workflow** deploying API and documentation to Kubernetes with automated rollout validation and rollback on failure.

<p align="right"><a href="#readme-top">Top ↑</a></p>

<!-- TECH STACK -->
## Tech Stack

* First technology
* Second technology
* ...

<p align="right"><a href="#readme-top">Top ↑</a></p>

<!-- LICENSE -->
## License
[MIT License](https://github.com/michelepatella/lstm-cache-eviction-policy/blob/main/LICENSE)

<p align="right"><a href="#readme-top">Top ↑</a></p>
