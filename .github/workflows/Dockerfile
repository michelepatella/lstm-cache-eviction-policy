FROM ubuntu:22.04

# Install essentials
RUN apt-get update && apt-get install -y \
    curl wget sudo git docker.io docker-compose build-essential \
    python3 python3-pip python3-venv unzip \
    libicu-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m runner
WORKDIR /home/runner

# Download GitHub Actions Runner
RUN curl -O -L https://github.com/actions/runner/releases/download/v2.330.0/actions-runner-linux-arm64-2.330.0.tar.gz \
    && tar xzf actions-runner-linux-arm64-2.330.0.tar.gz \
    && rm actions-runner-linux-arm64-2.330.0.tar.gz

# Start runner
COPY --chmod=755 entrypoint.sh /home/runner/entrypoint.sh
USER runner
ENTRYPOINT ["/home/runner/entrypoint.sh"]
