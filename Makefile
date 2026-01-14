include .env

DVC_LOCK_PATH := dvc.lock
DVC_FILES := "*.dvc"
GIT_IGNORE_FILE := .gitignore
LOCUST_PORT ?= ...
LOCUST_IMAGE_NAME ?= ...
GATEWAY_API_ENDPOINT_BASE_URL ?= ...
VC_COMMIT_MESSAGE := "dvc: Update tracked files"


# -------------------------------
# Locust
# -------------------------------

# Run a Locust Docker container
locust_run:
	docker run -it --rm \
	  -e TARGET_HOST=${GATEWAY_API_ENDPOINT_BASE_URL} \
	  -e LOCUST_PORT=${LOCUST_PORT} \
	  -p ${LOCUST_PORT}:${LOCUST_PORT} \
	  ${LOCUST_IMAGE_NAME}


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
