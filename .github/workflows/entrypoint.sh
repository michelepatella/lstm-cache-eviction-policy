#!/bin/bash
set -e

# Fix docker socket permissions
if [ -S /var/run/docker.sock ]; then
  sudo chmod 666 /var/run/docker.sock
fi

if [ ! -f .runner ]; then
  echo "Configuring runner..."
  ./config.sh --url "$RUNNER_REPO_URL" \
              --token "$RUNNER_TOKEN" \
              --name "${RUNNER_NAME:-docker-runner}" \
              --labels "${RUNNER_LABELS:-self-hosted,docker}" \
              --unattended \
              --replace
fi

echo "Starting runner..."
exec ./run.sh
