FROM python:3.12-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY src/api/services/scorer/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/src/api/services/scorer \
    /app/src/components/caches/implementations/items/scores \
    /app/src/components/logs/levels \
    /app/src/components/logs/levels/utils \
    /app/src/components/math \
    /app/src/api

# Copy all needed files
COPY src/api/services/scorer/scorer_service_grpc_server.py /app/src/api/services/scorer/
COPY src/api/services/scorer/scorer_service.py /app/src/api/services/scorer/
COPY src/api/services/scorer/scorer_service_pb2.py /app/src/api/services/scorer/
COPY src/api/services/scorer/scorer_service_pb2_grpc.py /app/src/api/services/scorer/
COPY src/api/const.py /app/src/api/
COPY src/components/caches/implementations/items/scores/survival_uncertainty_calculator.py /app/src/components/caches/implementations/items/scores/
COPY src/components/const.py /app/src/components/
COPY src/components/logs/levels/debug_logger.py /app/src/components/logs/levels/
COPY src/components/logs/levels/error_logger.py /app/src/components/logs/levels/
COPY src/components/logs/levels/utils/logger.py /app/src/components/logs/levels/utils
COPY src/components/math/confidence_interval_calculator.py /app/src/components/math/
COPY src/const.py /app/src/

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Set port
ARG SCORER_SERVICE_PORT
EXPOSE ${SCORER_SERVICE_PORT}

# Move to non-root user
USER appuser

# Run microservice
CMD ["python", "-m", "api.services.scorer.scorer_service_grpc_server"]
