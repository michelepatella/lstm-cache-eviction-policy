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
    <a href="https://dagshub.com/michelepatella/lstm-cache-eviction-policy">View DagsHub</a>
    &middot;
    <a href="https://dagshub.com/michelepatella/lstm-cache-eviction-policy.mlflow/">Explore MLflow</a>
    &middot;
    <a href="https://michelepatella.github.io/lstm-cache-eviction-policy/">Read Docs</a>
  </p>

  <!-- BADGES -->
  ![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![CI](https://img.shields.io/github/actions/workflow/status/michelepatella/lstm-cache-eviction-policy/ci.yml?logo=github-actions&style=for-the-badge&label=CI&logoColor=white)
  ![Coverage](https://img.shields.io/badge/Coverage-72%25-yellow?style=for-the-badge&logo=pytest&logoColor=white)
  ![GitHub License](https://img.shields.io/github/license/michelepatella/lstm-cache-eviction-policy?&style=for-the-badge&color=green&logo=github)
</div>

<br/>

<!-- THE CHALLENGE -->
## The Challenge
Traditional cache eviction policies such as LRU and LFU rely solely on past access patterns and cannot anticipate future demand. In dynamic, real-world workloads this often results in:

- Inefficient resource utilization  
- Increased infrastructure and operational costs  
- Degraded system performance and user experience

This project addresses these limitations by introducing a deep learning–driven cache eviction strategy.

<p align="right"><a href="#readme-top">Top ↑</a></p>

<!-- KEY FEATURES -->
## Key Features

* Scalable deep learning cache eviction policy using multi-step LSTM for adaptive eviction.  
* Uncertainty quantification via Monte Carlo dropout for confidence-aware eviction.  
* Temporal-aware loss function prioritizing imminent accesses for reducing cache misses.   
* Inference optimization via model quantization and weight pruning for faster execution.  
* Resilient cold-start handling with fallback and automated exclusion of frequent/recent keys for better stability.
    
<p align="right"><a href="#readme-top">Top ↑</a></p>

<!-- IMPACT -->
## Impact

Summarize results achieved by the proposed solution.

<p align="right"><a href="#readme-top">Top ↑</a></p>

<!-- SYSTEM ARCHITECTURE -->
## System Architecture

Show and describe the system architecture.

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
