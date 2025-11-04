# -------------------------------
# Constants
# -------------------------------

SRC_DIRECTORY := src
DOCS_OUTPUT_DIRECTORY := docs/_build/html

REQUIREMENTS_PATH := requirements.txt
DVC_LOCK_PATH := dvc.lock

VC_COMMIT_DVC_TARGET_DIRECTORY_PATTERNS := $(DVC_LOCK_PATH)
VC_COMMIT_MESSAGE := "dvc: Update tracked files"

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
# DVC
# -------------------------------

# View tracked file by DVC
dvc_list:
	dvc list .

# Update files tracked by DVC from remote
dvc_update:
	dvc pull
	dvc checkout

# Show DVC pipeline
dvc_pipeline_show:
	dvc dag

# Run the whole DVC pipeline
dvc_pipeline_run:
	dvc repro
	git add $(VC_COMMIT_DVC_TARGET_DIRECTORY_PATTERNS) .gitignore
	git commit -m $(VC_COMMIT_MESSAGE)
	dvc status
	dvc push
	git push

# Run a specific DVC pipeline stage
dvc_pipeline_stage_run:
	dvc repro $(STAGE_NAME)
	git add $(VC_COMMIT_DVC_TARGET_DIRECTORY_PATTERNS) .gitignore
	git commit -m $(VC_COMMIT_MESSAGE)
	dvc status
	dvc push
	git push

# Run the whole DVC pipeline (Force)
dvc_pipeline_run_force:
	dvc repro --force
	git add $(VC_COMMIT_DVC_TARGET_DIRECTORY_PATTERNS) .gitignore
	git commit -m $(VC_COMMIT_MESSAGE)
	dvc status
	dvc push
	git push

# Run a specific DVC pipeline stage (Force)
dvc_pipeline_stage_run_force:
	dvc repro $(STAGE_NAME) --force
	git add $(VC_COMMIT_DVC_TARGET_DIRECTORY_PATTERNS) .gitignore
	git commit -m $(VC_COMMIT_MESSAGE)
	dvc status
	dvc push
	git push

# Show metrics
dvc_metrics_show:
	dvc metrics show --json | jq

# Show metric differences
dvc_metrics_diff:
	dvc metrics diff --json | jq

# Show plots
dvc_plots_show:
	dvc plots show


# -------------------------------
# Quality Assurance
# -------------------------------

# Fix code
code_fix:
	ruff check $(SRC_DIRECTORY) --fix

# Format code
code_format:
	ruff format $(SRC_DIRECTORY)

# Lint code with Pylint
code_lint:
	pylint $(SRC_DIRECTORY)

# Check type on code
code_check_type:
	mypy $(SRC_DIRECTORY)


# -------------------------------
# Documentation
# -------------------------------

# Generate documentation
docs_generate:
	PYTHONPATH=$(shell pwd)/$(SRC_DIRECTORY) pdoc --html $(SRC_DIRECTORY)/ --output-dir $(DOCS_OUTPUT_DIRECTORY) --force


# -------------------------------
# Cleanup
# -------------------------------

# Clean pycache
pycache_clean:
	find . -type d -name $(PYCACHE_NAME) -exec rm -rf {} +

# Clean DVC cache
dvc_cache_clean:
	dvc gc -a