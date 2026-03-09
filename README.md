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

**Limitations of ML-Based Policies**  
ML-based policies provide predictive power but ignore model uncertainty.  
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
**Inception**
- **ML requirements** defining data, performance, and quality constraints.
- **AI governance self-assessment** classifying the system under the EU AI Act.
- **ML Canvas specification** formalizing objectives, stakeholders, and system constraints.

<!-- 'REPRODUCIBILITY' SUBSECTION -->
**Reproducibility**
- **Code versioning** with Git using a Git Flow-like branching strategy.  
- **Data and pipeline management** with DVC for reproducible pipelines and large artifact tracking.  
- **Experiment and model lifecycle management** with MLflow, including experiment tracking and model registry.

<!-- 'QUALITY ASSURANCE' SUBSECTION -->
**Quality Assurance**
- **Static analysis and pre-commit checks** enforcing code quality, security, formatting, and repository hygiene.  
- **Comprehensive testing** of code, data, models, and API with unit, integration, system, acceptance, and behavioral tests using pytest, Great Expectations, and DeepChecks. 
- **Production-safe deployment** with online canary testing and KPIs monitoring.

<!-- 'API' SUBSECTION -->
**API**
- **Deep learning microservice API** exposing RESTful endpoints and incapsulating the full ML pipeline via FastAPI.
- **API Gateway** orchestrating gRPC services with request validation, configuration, standardized responses, and monitoring.  
- **Featurizer, Predictor, Scorer, and Decider services** handling feature engineering, confidence-aware predictions, hybrid scoring, and top-K eviction selection.  
- **Documented artifacts** including model and dataset cards, plus OpenAPI specification.

<!-- 'DEPLOYMENT' SUBSECTION -->
**Deployment**
- **Containerization and compose** packaging microservices into isolated Docker containers and orchestrating them.  
- **Kubernetes and cloud deployment** orchestrating containers on a managed Google Cloud Kubernetes cluster with Artifact Registry.  
- **CI pipeline** automating static analysis, security scans, testing, multi-registry container builds, and docs generation. 
- **CD pipeline** deploying API with rollout validation, API tests, automated rollback, and docs publishing.

<!-- 'MONITORING' SUBSECTION -->
**Monitoring**
- **Availability monitoring** with global health checks and incident alerting via Better Uptime.
- **Load testing** simulating realistic traffic and failures with distributed Locust.
- **Observability stack** collecting metrics with Prometheus, alerting through Alertmanager, and visualized via Grafana.
- **Performance monitoring** detecting drifts and model degradation with Alibi Detect and Deepchecks pipelines.

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
