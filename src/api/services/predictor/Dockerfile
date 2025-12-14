FROM python:3.12-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r /app/requirements.txt

# NOTE: Install these libraries separately to
# avoid conflicts with other libraries
RUN pip install --no-cache-dir grpcio==1.76.0 grpcio-tools==1.76.0 -v

# Copy codes
COPY src/ /app/src/
ENV PYTHONPATH=/app/src

# Set port
ARG PREDICTOR_SERVICE_PORT
EXPOSE ${PREDICTOR_SERVICE_PORT}

# Move to non-root user
USER appuser

# Run microservice
CMD ["python", "-m", "src.api.services.predictor.predictor_service_grpc_server"]
