# -------------------------------
# Constants
# -------------------------------

# Dependencies
REQUIREMENTS_PATH := requirements.txt

# DVC (Data Versioning Control)
STATIC_RAW_DATASET_PATH := data/raw/static/static_raw_dataset.csv
DYNAMIC_RAW_DATASET_PATH := data/raw/dynamic/dynamic_raw_dataset.csv
STATIC_PROCESSED_DATASET_PATH := data/processed/static/static_processed_dataset.csv
DYNAMIC_PROCESSED_DATASET_PATH := data/processed/dynamic/dynamic_processed_dataset.csv
DATASETS_DIRECTORY_PATTERN := data/**/*.dvc
DATA_COMMIT_MESSAGE := "data: Update dataset(s)"

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
# DVC (Data Versioning Control)
# -------------------------------

# Check data status on DVC and Git
data_check_status:
	dvc status
	git status

# Add data to DVC
data_add:
	dvc add $(STATIC_RAW_DATASET_PATH)
	dvc add $(DYNAMIC_RAW_DATASET_PATH)
	dvc add $(STATIC_PROCESSED_DATASET_PATH)
	dvc add $(DYNAMIC_PROCESSED_DATASET_PATH)

# Commit data on Git
data_commit:
	git add $(DATASETS_DIRECTORY_PATTERN) .gitignore
	git commit -m $(DATA_COMMIT_MESSAGE)

# Push data to DVC and Git
data_push:
	dvc push
	git push

# Update and push data
data_update_and_push: data_add data_commit data_push

# Pull data from DVC
data_pull:
	dvc pull


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
	find . -name $(PYCACHE_NAME) -exec rm -rf {} +

# Clean DVC cache
dvc_cache_clean:
	dvc gc -a


# -------------------------------
# PHONY
# -------------------------------
.PHONY: deps_install deps_update \
	data_check_status data_add data_commit data_push data_update_and_push data_pull \
	code_format code_sort_imports code_fix_style code_check_lint code_check_type code_check_quality \
	docs_generate logs_clean pycache_clean dvc_cache_clean