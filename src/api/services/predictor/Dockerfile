FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src/ .

# Set port
ARG PREDICTOR_SERVICE_PORT
ENV PREDICTOR_SERVICE_PORT=${PREDICTOR_SERVICE_PORT}
EXPOSE ${PREDICTOR_SERVICE_PORT}

# Run microservice
CMD ["python", "-m", "api.services.predictor.predictor_service_grpc_server"]
