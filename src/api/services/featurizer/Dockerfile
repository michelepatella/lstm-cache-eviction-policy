FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src/ .

# Set port
ARG FEATURIZER_SERVICE_PORT
ENV FEATURIZER_SERVICE_PORT=${FEATURIZER_SERVICE_PORT}
EXPOSE ${FEATURIZER_SERVICE_PORT}

# Run microservice
CMD ["python", "-m", "api.services.featurizer.featurizer_service_grpc_server"]
