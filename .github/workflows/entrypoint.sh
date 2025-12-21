#!/bin/bash
set -e

if [ -z "$RUNNER_TOKEN" ]; then
  echo "RUNNER_TOKEN not set"
  exit 1
fi

if [ ! -f .runner ]; then
  echo "Configuring runner..."
  ./config.sh \
    --url "$RUNNER_REPO_URL" \
    --token "$RUNNER_TOKEN" \
    --name "${RUNNER_NAME:-docker-runner}" \
    --labels "${RUNNER_LABELS:-self-hosted,docker}" \
    --unattended \
    --replace
fi

echo "Starting runner..."
exec ./run.sh
