FROM python:3.12-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY src/api/gateway/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/src/api/gateway/callers \
    /app/src/api/config/pydantic/sections \
    /app/src/api/services/decider \
    /app/src/api/services/featurizer \
    /app/src/api/services/predictor \
    /app/src/api/services/scorer \
    /app/src/components/logs/handlers \
    /app/src/components/logs/levels \
    /app/src/components/yaml/io \
    /app/src/components/assertions \
    /app/src/api

# Copy all needed files
COPY src/api/gateway/main.py /app/src/api/gateway/main.py
COPY src/api/gateway/schemas.py /app/src/api/gateway/schemas.py
COPY src/api/gateway/callers/decider_service_caller.py /app/src/api/gateway/callers/decider_service_caller.py
COPY src/api/gateway/callers/featurizer_service_caller.py /app/src/api/gateway/callers/featurizer_service_caller.py
COPY src/api/gateway/callers/predictor_service_caller.py /app/src/api/gateway/callers/predictor_service_caller.py
COPY src/api/gateway/callers/scorer_service_caller.py /app/src/api/gateway/callers/scorer_service_caller.py
COPY src/api/config/pydantic/api_config.py /app/src/api/config/pydantic/api_config.py
COPY src/api/config/pydantic/sections/kwargs_api_config.py /app/src/api/config/pydantic/sections/kwargs_api_config.py
COPY src/api/config/pydantic/sections/logs_api_config.py /app/src/api/config/pydantic/sections/logs_api_config.py
COPY src/api/config/pydantic/sections/model_api_config.py /app/src/api/config/pydantic/sections/model_api_config.py
COPY src/api/config/pydantic/sections/resources_api_config.py /app/src/api/config/pydantic/sections/resources_api_config.py
COPY src/api/const.py /app/src/api/const.py
COPY src/api/services/decider/decider_service_pb2.py /app/src/api/services/decider/decider_service_pb2.py
COPY src/api/services/decider/decider_service_pb2_grpc.py /app/src/api/services/decider/decider_service_pb2_grpc.py
COPY src/api/services/featurizer/featurizer_service_pb2.py /app/src/api/services/featurizer/featurizer_service_pb2.py
COPY src/api/services/featurizer/featurizer_service_pb2_grpc.py /app/src/api/services/featurizer/featurizer_service_pb2_grpc.py
COPY src/api/services/predictor/predictor_service_pb2.py /app/src/api/services/predictor/predictor_service_pb2.py
COPY src/api/services/predictor/predictor_service_pb2_grpc.py /app/src/api/services/predictor/predictor_service_pb2_grpc.py
COPY src/api/services/scorer/scorer_service_pb2.py /app/src/api/services/scorer/scorer_service_pb2.py
COPY src/api/services/scorer/scorer_service_pb2_grpc.py /app/src/api/services/scorer/scorer_service_pb2_grpc.py
COPY src/components/assertions/choice_field_assertor.py /app/src/components/assertions/choice_field_assertor.py
COPY src/components/const.py /app/src/components/const.py
COPY src/components/logs/handlers/grafana_loki_handler.py /app/src/components/logs/handlers/grafana_loki_handler.py
COPY src/components/logs/initializer.py /app/src/components/logs/initializer.py
COPY src/components/logs/levels/debug_logger.py /app/src/components/logs/levels/debug_logger.py
COPY src/components/logs/levels/error_logger.py /app/src/components/logs/levels/error_logger.py
COPY src/components/logs/levels/info_logger.py /app/src/components/logs/levels/info_logger.py
COPY src/components/logs/levels/utils/logger.py /app/src/components/logs/levels/utils/logger.py
COPY src/components/yaml/io/loader.py /app/src/components/yaml/io/loader.py
COPY src/const.py /app/src/
COPY src/api/config/config.yaml /app/src/api/config/config.yaml

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Set port
ARG GATEWAY_API_PORT
ENV GATEWAY_API_PORT=${GATEWAY_API_PORT}
EXPOSE ${GATEWAY_API_PORT}

# Move to non-root user
USER appuser

# Run API Gateway
CMD sh -c "uvicorn api.gateway.main:app --host 0.0.0.0 --port ${GATEWAY_API_PORT}"
