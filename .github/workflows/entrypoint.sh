#!/bin/bash
set -e

# Fix Docker socket permissions
if [ -S "$DOCKER_SOCKET" ]; then
  sudo chmod 666 "$DOCKER_SOCKET"
fi

# Configuration
if [ ! -f .runner ]; then
  ./config.sh --url "${RUNNER_REPO_URL}" \
              --token "${RUNNER_TOKEN}" \
              --name "${RUNNER_NAME}" \
              --labels "${RUNNER_LABELS}" \
              --unattended \
              --replace
fi

# Run the runner
exec "$RUNNER_SCRIPT"
