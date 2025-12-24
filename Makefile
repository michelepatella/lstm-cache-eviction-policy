DVC_LOCK_PATH := dvc.lock
DVC_FILES := "*.dvc"

GIT_IGNORE_FILE := .gitignore

VC_COMMIT_MESSAGE := "dvc: Update tracked files"

PYCACHE_NAME := "__pycache__"


# -------------------------------
# Clean
# -------------------------------

# Clean pycache
clean_pycache:
	find . -type d -name $(PYCACHE_NAME) -exec rm -rf {} +

# Clean DVC cache
clean_dvc_cache:
	dvc gc -a

# -------------------------------
# DVC
# -------------------------------

# Update files tracked by DVC from remote
dvc_update:
	dvc pull
	dvc checkout

# Run the whole DVC pipeline
dvc_pipeline_run:
	dvc repro
	git add $(shell find . -name $(DVC_FILES))
	git add $(DVC_LOCK_PATH) $(GIT_IGNORE_FILE)
	git commit -m $(VC_COMMIT_MESSAGE) --no-verify || true
	dvc status
	dvc push
	git push

# Run a specific DVC pipeline stage
dvc_pipeline_stage_run:
	dvc repro $(STAGE_NAME)
	git add $(shell find . -name $(DVC_FILES))
	git add $(DVC_LOCK_PATH) $(GIT_IGNORE_FILE)
	git commit -m $(VC_COMMIT_MESSAGE) --no-verify || true
	dvc status
	dvc push
	git push

# Run the whole DVC pipeline (Force)
dvc_pipeline_run_force:
	dvc repro --force
	git add $(shell find . -name $(DVC_FILES))
	git add $(DVC_LOCK_PATH) $(GIT_IGNORE_FILE)
	git commit -m $(VC_COMMIT_MESSAGE) --no-verify || true
	dvc status
	dvc push
	git push

# Run a specific DVC pipeline stage (Force)
dvc_pipeline_stage_run_force:
	dvc repro $(STAGE_NAME) --force
	git add $(shell find . -name $(DVC_FILES))
	git add $(DVC_LOCK_PATH) $(GIT_IGNORE_FILE)
	git commit -m $(VC_COMMIT_MESSAGE) --no-verify || true
	dvc status
	dvc push
	git push

# Show DVC pipeline
dvc_pipeline_show:
	dvc dag

# Show metrics
dvc_metrics_show:
	dvc metrics show --json | jq

# Show metric differences
dvc_metrics_diff:
	dvc metrics diff --json | jq
