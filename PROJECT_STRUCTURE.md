```text
.
├── PROJECT_STRUCTURE.md                 <- This file
├── LICENSE.md                           <- Project License
├── README.md                            <- Project overview
├── .dvc                                 <- DVC metadata and cache
├── .github                              <- GitHub workflows
├── .dvcignore                           <- Files/folders ignored by DVC
├── .env                                 <- Environment variables
├── .gitignore                           <- Files/folders ignored by Git
├── .pre-commit-config.yaml              <- Pre-commit hooks config
├── Makefile                             <- Makefile for most widely used commands
├── cloudbuild.yaml                      <- Google Cloud build config for Docker images
├── data
│   ├── README.md                        <- Dataset card
│   ├── processed                        <- Final, processed datasets
│   └── raw                              <- Original datasets
├── docker-compose.yml                   <- Multi-service orchestration
├── docs
│   ├── _build                           <- Generated docs output
│   └── ml_canvas.pdf                    <- ML Canvas for system requirements
├── dvc.lock                             <- Locked pipeline state
├── dvc.yaml                             <- Pipeline definition and orchestration
├── k8s                                  <- k8s manifests
├── load_test                            <- Locust API tests
├── models                               <- Trained, optimized models and their card
├── notebooks                            <- Jupyter notebooks
├── params.yaml                          <- Pipeline params for DVC
├── prometheus-grafana                   <- Monitoring and alerting setup
├── pyproject.toml                       <- Project config
├── reports                              <- Generated analysis as PDF, HTML, png, etc.
│   ├── figures                          <- Generated figures
│   ├── monitoring                       <- Generated monitoring reports
│   └── tests                            <- Generated test reports
├── requirements-dev.txt                 <- Development project requirements
├── requirements.txt                     <- General project requirements
├── retraining_checkpoint.json           <- Last completed model training checkpoint
├── src                                  <- Project source code
│   ├── api                              <- API implementation
│   ├── components                       <- Shared and utility components
│   ├── const.py                         <- Project-wide constants
│   └── pipeline                         <- ML pipeline implementation
└── tests                                <- Tests for code, data, models, and API
