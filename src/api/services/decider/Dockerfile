FROM python:3.12-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY src/api/services/decider/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/src/api \
    /app/src/api/services/decider \
    /app/src/components/caches/implementations/items/evictions \
    /app/src/components/logs/levels/utils \
    /app/src/components/logs/levels/ \
    /app/src/components/caches/implementations/items/evictions \
    /app/src/api/services \
    && touch /app/src/api/services/.placeholder

# Copy all needed files
COPY src/const.py /app/src/
COPY src/api/const.py /app/src/api/
COPY src/api/services/decider/decider_service.py /app/src/api/services/decider/
COPY src/api/services/decider/decider_service_grpc_server.py /app/src/api/services/decider/
COPY src/api/services/decider/decider_service_pb2.py /app/src/api/services/decider/
COPY src/api/services/decider/decider_service_pb2_grpc.py /app/src/api/services/decider/
COPY src/components/caches/implementations/items/evictions/score_based_evictor.py /app/src/components/caches/implementations/items/evictions/
COPY src/components/logs/levels/debug_logger.py /app/src/components/logs/levels/
COPY src/components/logs/levels/error_logger.py /app/src/components/logs/levels/
COPY src/components/logs/levels/info_logger.py /app/src/components/logs/levels/
COPY src/components/logs/levels/utils/logger.py /app/src/components/logs/levels/utils/
COPY src/components/json/io/loader.py /app/src/components/json/io/loader.py
COPY src/components/json/io/saver.py /app/src/components/json/io/saver.py

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Set port
ARG DECIDER_SERVICE_PORT
EXPOSE ${DECIDER_SERVICE_PORT}

# Move to non-root user
USER appuser

# Run microservice
CMD ["python", "-m", "api.services.decider.decider_service_grpc_server"]
