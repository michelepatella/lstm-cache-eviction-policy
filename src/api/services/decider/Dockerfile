FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy codes
COPY src/ /app/src/
ENV PYTHONPATH=/app/src

# Set port
ARG DECIDER_SERVICE_PORT
ENV DECIDER_SERVICE_PORT=${DECIDER_SERVICE_PORT}
EXPOSE ${DECIDER_SERVICE_PORT}

# Run microservice
CMD ["python", "-m", "api.services.decider.decider_service_grpc_server"]
