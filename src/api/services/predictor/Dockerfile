FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Install Git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set Git token arg
ARG GIT_HUB_TOKEN

# Clone repo
RUN git clone --branch feature/docker https://$GIT_HUB_TOKEN@github.com/michelepatella/lstm-cache-eviction-policy.git /app

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir grpcio==1.76.0 grpcio-tools==1.76.0 -v

# Set port
ARG PREDICTOR_SERVICE_PORT
ENV PREDICTOR_SERVICE_PORT=${PREDICTOR_SERVICE_PORT}
EXPOSE ${PREDICTOR_SERVICE_PORT}

# Run microservice
ENV PYTHONPATH=/app:$PYTHONPATH
CMD ["python", "-m", "src.api.services.predictor.predictor_service_grpc_server"]
