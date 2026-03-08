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
    <a href="https://dagshub.com/michelepatella/lstm-cache-eviction-policy">DagsHub Project</a>
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

<!-- 'THE PROBLEM' SECTION -->
## The Problem
**Traditional cache eviction policies** rely only on past access patterns and cannot anticipate future demand.  
→ _Inefficient resource utilization_  
→ _Increased infrastructure and operational costs_  
→ _Degraded system performance and user experience_

**Machine/deep learning-based cache eviction policies** ignore model uncertainty.  
→ _Unreliable decisions_  

**This project** addresses these limitations by introducing an uncertainty-aware, deep learning–driven cache eviction strategy.

<p align="right"><a href="#readme-top">Top ↑</a></p>

<!-- 'KEY FEATURES' SECTION -->
## Key Features

### Core
- **Adaptive eviction** with autoregressive, multi-step LSTM capturing future data access patterns.
- **Uncertainty-aware scoring** protecting high-uncertainty data items using Monte Carlo dropout.
- **Modular architecture** separating featurization, prediction, scoring, and eviction logic.
- **Feature engineering** with cyclical time encoding and local recency/frequency for temporal locality.
- **Temporal-aware loss** prioritizing imminent-key predictions to minimize cache misses.
- **Optimized training** with hyperparameter search, early stopping, and parallel execution via Ray and PyTorch DDP.
- **Optimized inference** via 8-bit quantization and 20% weight pruning for faster CPU execution.

### MLOps

#### Inception
- **ML requirements engineering** defining data, performance, and quality requirements.
- **AI governance self-assessment** to classify the system under the EU AI Act.
- **System specification** through ML Canvas to formalize objectives, stakeholders, and system constraints.

#### Reproducibility
- **Code versioning** with Git using a Git Flow-like branching strategy and pull requests-based integration.
- **Reproducible ML pipeline** defined with DVC for tracking pipeline stages.
- **Data and model versioning** with DVC to keep large artifacts outside Git.
- **Experiment tracking** with MLflow, logging parameters, metrics, and artifacts.
- **Model lifecycle management** through MLflow Model Registry, versioning and organizing production and staging models.

<p align="right"><a href="#readme-top">Top ↑</a></p>

<!-- 'SYSTEM ARCHITECTURE' SECTION -->
## System Architecture

Show and describe the system architecture.

<p align="right"><a href="#readme-top">Top ↑</a></p>

<!-- 'IMPACT' SECTION -->
## Impact

Summarize results achieved by the proposed solution.

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
