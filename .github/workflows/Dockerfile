FROM ubuntu:22.04

# Install essentials
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils \
    python3-pip \
    bash jq curl wget git docker.io unzip \
    build-essential libicu-dev \
    ca-certificates apt-transport-https lsb-release gnupg sudo \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Create runner user and Docker group
RUN groupadd -r runner && useradd -m -r -g runner runner \
    && groupadd -g 999 docker || true \
    && usermod -aG docker runner \
    && echo "runner ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
WORKDIR /home/runner

# Install Docker
RUN curl -fsSL https://get.docker.com | sh

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Install kubectl
ENV KUBECTL_VERSION=1.28.0
RUN curl -LO "https://dl.k8s.io/release/v${KUBECTL_VERSION}/bin/linux/arm64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/

# Install Kustomize
ENV KUSTOMIZE_VERSION=5.8.0
RUN curl -sLo kustomize.tar.gz https://sourceforge.net/projects/kustomize.mirror/files/kustomize_v${KUSTOMIZE_VERSION}/kustomize_v${KUSTOMIZE_VERSION}_linux_arm64.tar.gz \
    && tar -xzf kustomize.tar.gz kustomize \
    && chmod +x kustomize \
    && mv kustomize /usr/local/bin/ \
    && rm kustomize.tar.gz

# Install Hadolint
ENV HADOLINT_VERSION=2.12.0
RUN wget -O /usr/local/bin/hadolint https://github.com/hadolint/hadolint/releases/download/v${HADOLINT_VERSION}/hadolint-Linux-arm64 \
    && chmod +x /usr/local/bin/hadolint

# Install Kube-score
ENV KUBESCORE_VERSION=1.19.0
RUN wget -O /usr/local/bin/kube-score https://github.com/zegl/kube-score/releases/download/v${KUBESCORE_VERSION}/kube-score_${KUBESCORE_VERSION}_linux_arm64 \
    && chmod +x /usr/local/bin/kube-score

# Install Trivy
ENV TRIVY_VERSION=0.68.2
RUN curl -sLo trivy.tar.gz https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_Linux-ARM64.tar.gz \
    && tar -xzf trivy.tar.gz trivy \
    && mv trivy /usr/local/bin/ \
    && rm trivy.tar.gz

# Install yq
ENV YQ_VERSION=4.45.1
RUN wget https://github.com/mikefarah/yq/releases/download/v${YQ_VERSION}/yq_linux_amd64 -O /usr/local/bin/yq \
    && chmod +x /usr/local/bin/yq

# Install GitHub Actions Runner
ENV RUNNER_VERSION=2.321.0
RUN curl -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz \
    && tar xzf actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz \
    && rm actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz

# Copy and install Python requirements
COPY requirements.txt requirements-dev.txt /home/runner/
RUN chown runner:runner /home/runner/requirements* \
    && python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir -r /home/runner/requirements.txt \
    && python3 -m pip install --no-cache-dir -r /home/runner/requirements-dev.txt \
    && python3 -m pip uninstall -y botocore boto3 \
    && python3 -m pip install --no-cache-dir "dvc[s3]>=3.0.0" boto3 botocore

# Start runner
COPY --chmod=755 .github/workflows/entrypoint.sh /home/runner/entrypoint.sh
RUN chown -R runner:runner /home/runner
USER runner
ENTRYPOINT ["/home/runner/entrypoint.sh"]
