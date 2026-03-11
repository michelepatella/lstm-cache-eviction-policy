<!-- TOP -->
<a id="readme-top"></a>

<!-- HEADER -->
<div align="center">

  <!-- TITLE -->
  <h1 align="center">LSTM Cache Eviction Policy</h1>

  <!-- DESCRIPTION AND LINKS -->
  <p align="center">
    Deep learning-driven cache eviction policy using LSTM forecasting and uncertainty estimation, deployed as a production microservice with a full MLOps pipeline.
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

<div align="center">
  <img width="3019" height="1340" alt="image" src="https://github.com/user-attachments/assets/7dec855d-c673-482b-98a2-b056bfe60d38" />
  <p align="center">
  Integration of the LSTM cache eviction policy within a standard cache-aside architecture, separating the synchronous critical path (orange) from the asynchronous path (blue). <br>
  </p>
</div>

<br/>

<!-- TABLE OF CONTENT -->
<details>
  <summary><strong>Table of Contents</strong></summary>

- [Motivation & Impact](#motivation--impact)
- [Features](#features)
  - [ML Core](#ml-core)
  - [MLOps](#mlops)
- [Tech Stack](#tech-stack)
- [Credits](#credits)
- [License](#license)

</details>

<br/>

<!-- 'MOTIVATION & IMPACT' SECTION -->
## Motivation & Impact
**Limitations of Traditional Policies**  
Traditional cache eviction policies such as LRU and LFU rely only on past access patterns and cannot anticipate future demand.  
→ _**Inefficient** resource utilization_  
→ _**Higher** infrastructure costs_  
→ _**Degraded** system performance_

**Limitations of ML-Based Policies**  
ML-based policies can predict access patterns but typically ignore model uncertainty.  
→ _**Unreliable** eviction decisions_  

**The Proposed Solution**  
This project introduces a **deep learning-based, uncertainty-aware cache eviction policy**, achieving:  
* **+14%** hit rate vs LRU
* **+6%** hit rate vs LFU
* **+1%** hit rate vs uncertainty-unaware policy

<p align="right"><a href="#readme-top">Top ↑</a></p>

<!-- 'FEATURES' SECTION -->
## Features

<!-- 'ML CORE' SUBSECTION -->
### ML Core
- **Adaptive LSTM Eviction**: Autoregressive, multi-step predictions of future access patterns.
- **Uncertainty-Aware Scoring**: Protects high-uncertainty data items using Monte Carlo dropout.
- **Feature Engineering**: Captures temporal locality with cyclical time encoding and recency/frequency signals.
- **Temporal-Aware Loss**: Prioritizes near-future keys to minimize cache misses.
- **Optimized Training & Inference**: Distributed hyperparameter search and training with Ray and PyTorch DDP; quantization and pruning for faster CPU inference.

<!-- 'MLOPS' SUBSECTION -->
### MLOps

- **Inception**: ML requirements with ML Canvas and EU AI Act self-assessment.
- **Reproducibility**: Git with Git Flow for code versioning, DVC for artifacts and pipelines, and MLflow for experiments and model lifecycle management.
- **Quality Assurance**: Static analysis and pre-commit checks; comprehensive testing of code, data, models, and API with pytest, Great Expectations, and Deepchecks; production-safe deployment via online canary testing.
- **API**: RESTful endpoint via FastAPI and gRPC microservices, encapsulating feature engineering → confidence-aware predictions → hybrid scoring → top-K eviction selection.
- **Deployment**: Containerized microservices with Docker/Docker Compose; Kubernetes on Google Cloud Platform with Artifact Registry for hosting; automated CI/CD pipelines for static analysis, security scans, testing, building, and safe deployment.
- **Monitoring**: Availability checks and incident alerting with Better Uptime; load testing with Locust; metrics collection, visualization, and alerting with Prometheus, Grafana, and Alertmanager; performance and drift monitoring with Alibi Detect and Deepchecks.

<p align="right"><a href="#readme-top">Top ↑</a></p>

<!-- TECH STACK -->
## Tech Stack

* **Language**: Python 3.12+
* **ML & Data**: PyTorch, Scikit-learn, Ray, NumPy, Pandas, Pydantic
* **Reproducibility**: Git, DVC (with S3-compatible remote storage), MLflow, DagsHub
* **Quality Assurance**:
  * **Integrity & Workflow**: pre-commit, nbstripout
  * **Quality & Formatting**: Pylint, Pynblint, Hadolint, Ruff, kube-score
  * **Security**: Bandit, pip-audit, Trivy, Checkov
  * **Testing**: Pytest, Deepchecks, Great Expectations
* **API**: FastAPI, Uvicorn, gRPC
* **Deployment**:
  * **Containerization**: Docker, Docker Compose
  * **Orchestration**: Kubernetes
  * **Cloud Infrastructure**: Google Cloud Platform
  * **Infrastructure as Code**: Terraform, Kustomize
  * **CI/CD**: GitHub Actions
  * **Container Registries**: Google Artifact Registry, Docker Hub, GitHub Container Registry
* **Monitoring**:
  * **Resource-Level**: Better Uptime, Locust, Prometheus, Alertmanager, Grafana
  * **Performance-Level**: Alibi Detect, Deepchecks
* **Logging**: Grafana Loki, Structlog
* **Docs**: GitBook, pdoc, GitHub Pages, OpenAPI

<p align="right"><a href="#readme-top">Top ↑</a></p>

<!-- 'CREDITS' SECTION  -->
## Credits
Icons appearing in the figure above were created by Freepik and littleicon, available on https://www.flaticon.com.

<p align="right"><a href="#readme-top">Top ↑</a></p>

<!-- 'LICENSE' SECTION  -->
## License
[MIT License](https://github.com/michelepatella/lstm-cache-eviction-policy/blob/main/LICENSE)

<p align="right"><a href="#readme-top">Top ↑</a></p>
