FROM python:3.12-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codes
COPY src/ /app/src/
ENV PYTHONPATH=/app/src

# Set port
ARG FEATURIZER_SERVICE_PORT
EXPOSE ${FEATURIZER_SERVICE_PORT}

# Move to non-root user
USER appuser

# Run microservice
CMD ["python", "-m", "api.services.featurizer.featurizer_service_grpc_server"]
