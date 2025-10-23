# -------------------------------
# Constants
# -------------------------------

# Dependencies
REQUIREMENTS_PATH := requirements.txt

# VC / DVC
STATIC_RAW_DATASET_PATH := data/raw/static/static_raw_dataset.csv
DYNAMIC_RAW_DATASET_PATH := data/raw/dynamic/dynamic_raw_dataset.csv
STATIC_PROCESSED_DATASET_PATH := data/processed/static/static_processed_dataset.csv
DYNAMIC_PROCESSED_DATASET_PATH := data/processed/dynamic/dynamic_processed_dataset.csv

STATIC_MODEL_PATH := models/static/trained_static_model.pt
DYNAMIC_MODEL_PATH := models/dynamic/trained_dynamic_model.pt

STATIC_MODEL_RESULTS_PATH := reports/results/static/static_model_results.json
DYNAMIC_MODEL_RESULTS_PATH := reports/results/dynamic/dynamic_model_results.json
STATIC_SIMULATION_RESULTS_PATH := reports/results/static/static_simulations_results.json
DYNAMIC_SIMULATION_RESULTS_PATH := reports/results/dynamic/dynamic_simulations_results.json

STATIC_DAILY_PROFILE_PLOT_PATH := reports/plots/static/static_daily_profile.png
DYNAMIC_DAILY_PROFILE_PLOT_PATH := reports/plots/dynamic/dynamic_daily_profile.png
STATIC_KEY_USAGE_PLOT_PATH := reports/plots/static/static_key_usage.png
DYNAMIC_KEY_USAGE_PLOT_PATH := reports/plots/dynamic/dynamic_key_usage.png
STATIC_ZIPF_LOG_LOG_PLOT_PATH := reports/plots/static/static_zipf_log_log.png
DYNAMIC_ZIPF_LOG_LOG_PLOT_PATH := reports/plots/dynamic/dynamic_zipf_log_log.png
STATIC_HIT_MISS_RATES_PLOT_PATH := reports/plots/static/static_hit_miss_rates.png
DYNAMIC_HIT_MISS_RATES_PLOT_PATH := reports/plots/dynamic/dynamic_hit_miss_rates.png

DVC_TARGET_PATHS := $(STATIC_RAW_DATASET_PATH) $(DYNAMIC_RAW_DATASET_PATH) \
                $(STATIC_PROCESSED_DATASET_PATH) $(DYNAMIC_PROCESSED_DATASET_PATH) \
                $(STATIC_MODEL_PATH) $(DYNAMIC_MODEL_PATH) $(STATIC_MODEL_RESULTS_PATH) \
                $(DYNAMIC_MODEL_RESULTS_PATH) $(STATIC_SIMULATION_RESULTS_PATH) \
                $(DYNAMIC_SIMULATION_RESULTS_PATH) $(STATIC_DAILY_PROFILE_PLOT_PATH) \
                $(DYNAMIC_DAILY_PROFILE_PLOT_PATH) $(STATIC_KEY_USAGE_PLOT_PATH) \
                $(DYNAMIC_KEY_USAGE_PLOT_PATH) $(STATIC_ZIPF_LOG_LOG_PLOT_PATH) \
                $(DYNAMIC_ZIPF_LOG_LOG_PLOT_PATH)

DATASETS_DVC_DIRECTORY_PATTERN := data/**/*.dvc
MODELS_DVC_DIRECTORY_PATTERN := models/**/*.dvc
RESULTS_DVC_DIRECTORY_PATTERN := reports/**/*.dvc

VC_COMMIT_DVC_TARGET_DIRECTORY_PATTERNS := $(DATASETS_DVC_DIRECTORY_PATTERN) $(MODELS_DVC_DIRECTORY_PATTERN) $(RESULTS_DVC_DIRECTORY_PATTERN)
VC_COMMIT_MESSAGE := "dvc: Update tracked files"

# Quality Assurance & Documentation
SRC_DIRECTORY := src
DOCS_OUTPUT_DIRECTORY := docs/_build/html

# Cleanup
LOGS_DIRECTORY_PATTERN := logs/*/*.log
LOGS_DIRECTORY_PATTERN_ROTATED := logs/*/*.log.*
PYCACHE_NAME := "__pycache__"


# -------------------------------
# Dependencies
# -------------------------------

# Install dependencies specified by requirements.txt
deps_install:
	pip install -r $(REQUIREMENTS_PATH)

# Update dependencies in requirements.txt
deps_update:
	pip freeze > $(REQUIREMENTS_PATH)


# -------------------------------
# DVC / VC
# -------------------------------

# Pipeline for tracked files:
# (1) Check DVC / VC status
dvc_vc_check_status:
	dvc status
	git status

# (2) Add to DVC
dvc_add:
	dvc add $(DVC_TARGET_PATHS)

# (3) Commit to VC
vc_commit:
	git add $(VC_COMMIT_DVC_TARGET_DIRECTORY_PATTERNS) .gitignore
	git commit -m $(VC_COMMIT_MESSAGE)

# (4) Push to DVC / VC
dvc_vc_push:
	dvc push
	git push

# (1)-(5) Pipeline for DVC tracked files
dvc_vc_update_and_push: dvc_vc_check_status dvc_add vc_commit dvc_vc_push

# (DVC) Others:
# List DVC remotes
dvc_remote_list:
	dvc remote list

# List DVC-tracked files
dvc_file_list:
	dvc list .

# Remove tracked files from DVC
dvc_file_remove:
	dvc remove $(PATH)

# Remove DVC remote
dvc_remote_remove:
	dvc remote remove $(REMOTE_NAME)

# Pull latest files from DVC
dvc_pull:
	dvc pull

# Pull and checkout specific version
dvc_pull_version:
	dvc checkout $(VERSION)
	dvc pull

# Restore DVC-tracked files
dvc_checkout:
	dvc checkout

# Restore specific DVC-tracked file
dvc_checkout_file:
	dvc checkout $(PATH)

# Show DVC pipeline
dvc_pipeline_show:
	dvc dag

# Run the whole DVC pipeline
dvc_pipeline_run:
	dvc repro

# Run a specific pipeline stage
dvc_pipeline_stage_run:
	dvc repro $(STAGE_NAME)

# Run the whole DVC pipeline (Force)
dvc_pipeline_run_force:
	dvc repro --force

# Run a specific pipeline stage (Force)
dvc_pipeline_stage_run_force:
	dvc repro $(STAGE_NAME) --force

# Show metrics
dvc_metrics_show:
	dvc metrics show --json | jq

# Show metric differences
dvc_metrics_diff:
	dvc metrics diff --json | jq

# Show plots
dvc_plots_show:
	dvc plots show

# (VC) Others:
# Checkout VC files via version tag
vc_tag_checkout:
	git checkout $(TAG_NAME)

# Checkout specific VC file via version tag
vc_tag_checkout_file:
	git checkout $(TAG_NAME) $(PATH)

# Add a version tag in VC
vc_tag_create:
	git tag -a $(TAG_NAME) -m "$(TAG_MESSAGE)"

# Push a version tag to VC
vc_tag_push:
	git push origin $(TAG_NAME)

# Delete a local version tag from VC
vc_tag_delete_local:
	git tag -d $(TAG_NAME)

# Delete a remote version tag from VC
vc_tag_delete_remote:
	git push --delete origin $(TAG_NAME)


# -------------------------------
# Quality Assurance
# -------------------------------

# Format code
code_format:
	black $(SRC_DIRECTORY)

# Sort imports in code
code_sort_imports:
	isort $(SRC_DIRECTORY)

# Fix code style (formatting + import sorting)
code_fix_style: code_format code_sort_imports

# Check lint on code
code_check_lint:
	flake8 $(SRC_DIRECTORY)

# Check type on code
code_check_type:
	mypy $(SRC_DIRECTORY)

# Check code quality (linting + typing)
code_check_quality: code_check_lint code_check_type


# -------------------------------
# Documentation
# -------------------------------

# Generate documentation
docs_generate:
	PYTHONPATH=$(shell pwd)/$(SRC_DIRECTORY) pdoc --html $(SRC_DIRECTORY)/ --output-dir $(DOCS_OUTPUT_DIRECTORY) --force


# -------------------------------
# Cleanup
# -------------------------------

# Clean logs
logs_clean:
	rm -rf $(LOGS_DIRECTORY_PATTERN)
	rm -rf $(LOGS_DIRECTORY_PATTERN_ROTATED)

# Clean pycache
pycache_clean:
	find . -type d -name $(PYCACHE_NAME) -exec rm -rf {} +

# Clean DVC cache
dvc_cache_clean:
	dvc gc -a