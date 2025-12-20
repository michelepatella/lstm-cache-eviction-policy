include .env

ROOT_DIRECTORY := .
SRC_DIRECTORY := src
DOCS_OUTPUT_DIRECTORY := docs/_build/html

DVC_LOCK_PATH := dvc.lock
VC_COMMIT_MESSAGE := "dvc: Update tracked files"

PYCACHE_NAME := "__pycache__"

DOCKER_IMAGES_TAG=1.0.0


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
# Code
# -------------------------------

# Fix code
code_fix:
	ruff check $(ROOT_DIRECTORY) --fix

# Format code
code_format:
	ruff format $(ROOT_DIRECTORY)

# Lint code with Pylint
code_lint:
	pylint $(SRC_DIRECTORY)


# -------------------------------
# Docker
# -------------------------------

# Build containers
docker_build:
	docker-compose build --no-cache

# Make up containers
docker_up:
	docker-compose up --no-cache

# Restart containers
docker_restart:
	docker-compose restart

# Build and make up containers
docker_build_up:
	docker-compose build --no-cache && docker-compose up

# Push all the images to DockerHub
# and Github Packages
docker_push:
	@set -o allexport; . $(ROOT_DIRECTORY)/.env; set +o allexport; \
	for svc in $$(echo $$API_MICROSERVICES | tr ',' ' '); do \
		docker tag $$svc:latest $$DOCKER_HUB_USER/$$DOCKER_HUB_REPO:$$svc-$$DOCKER_IMAGES_TAG; \
		docker tag $$svc:latest $$GIT_HUB_REGISTRY/$$GIT_HUB_USER/$$GIT_HUB_REPO:$$svc-$$DOCKER_IMAGES_TAG; \
		docker push $$DOCKER_HUB_USER/$$DOCKER_HUB_REPO:$$svc-$$DOCKER_IMAGES_TAG; \
		docker push $$GIT_HUB_REGISTRY/$$GIT_HUB_USER/$$GIT_HUB_REPO:$$svc-$$DOCKER_IMAGES_TAG; \
	done


# -------------------------------
# Docs
# -------------------------------

# Generate documentation
docs_generate:
	PYTHONPATH=$(shell pwd)/$(SRC_DIRECTORY) pdoc --html $(SRC_DIRECTORY)/ --output-dir $(DOCS_OUTPUT_DIRECTORY) --force


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
	git add $(DVC_LOCK_PATH) .gitignore
	git commit -m $(VC_COMMIT_MESSAGE)
	dvc status
	dvc push
	git push

# Run a specific DVC pipeline stage
dvc_pipeline_stage_run:
	dvc repro $(STAGE_NAME)
	git add $(DVC_LOCK_PATH) .gitignore
	git commit -m $(VC_COMMIT_MESSAGE)
	dvc status
	dvc push
	git push

# Run the whole DVC pipeline (Force)
dvc_pipeline_run_force:
	dvc repro --force
	git add $(DVC_LOCK_PATH) .gitignore
	git commit -m $(VC_COMMIT_MESSAGE)
	dvc status
	dvc push
	git push

# Run a specific DVC pipeline stage (Force)
dvc_pipeline_stage_run_force:
	dvc repro $(STAGE_NAME) --force
	git add $(DVC_LOCK_PATH) .gitignore
	git commit -m $(VC_COMMIT_MESSAGE)
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


# -------------------------------
# Kubernetes (k8s)
# -------------------------------

# Apply k8s files (deployment, service, config, and secrets)
k8s_apply:
	@for dep in $$API_DEPLOYMENTS; do \
		kubectl apply -f $(ROOT_DIRECTORY)/k8s/$$dep; \
	done

# Force deployment rollouts
k8s_rollout:
	@for dep in $$API_DEPLOYMENTS; do \
		kubectl rollout restart deployment $$dep; \
	done
