FROM python:3.12-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY src/api/services/predictor/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# NOTE: Install these libraries separately to
# avoid conflicts with other libraries
RUN pip install --no-cache-dir grpcio==1.76.0 grpcio-tools==1.76.0 -v

# Create necessary directories
RUN mkdir -p /app/src/api/services/predictor \
    /app/src/api/config/pydantic/sections \
    /app/src/components/assertions \
    /app/src/components/backpropagation/core \
    /app/src/components/backpropagation/mc_dropout \
    /app/src/components/dataset/features/local \
    /app/src/components/device \
    /app/src/components/inference/autoregressive_rollout \
    /app/src/components/logs/levels \
    /app/src/components/logs/levels/utils \
    /app/src/components/loss \
    /app/src/components/model/mode \
    /app/src/components/time/transforms \
    /app/src/components/yaml/io \
    /app/src/api

# Copy all needed files
COPY src/api/services/predictor/predictor_service_grpc_server.py /app/src/api/services/predictor/
COPY src/api/services/predictor/predictor_service.py /app/src/api/services/predictor/
COPY src/api/services/predictor/predictor_service_pb2.py /app/src/api/services/predictor/
COPY src/api/services/predictor/predictor_service_pb2_grpc.py /app/src/api/services/predictor/
COPY src/api/const.py /app/src/api/
COPY src/api/config/pydantic/api_config.py /app/src/api/config/pydantic/api_config.py
COPY src/api/config/pydantic/sections/canary_tests_api_config.py /app/src/api/config/pydantic/sections/
COPY src/api/config/pydantic/sections/kwargs_api_config.py /app/src/api/config/pydantic/sections/
COPY src/api/config/pydantic/sections/logs_api_config.py /app/src/api/config/pydantic/sections/
COPY src/api/config/pydantic/sections/model_api_config.py /app/src/api/config/pydantic/sections/
COPY src/api/config/pydantic/sections/resources_api_config.py /app/src/api/config/pydantic/sections/
COPY src/api/config/config.yaml /app/src/api/config/config.yaml
COPY src/components/assertions/choice_field_assertor.py /app/src/components/assertions/
COPY src/components/backpropagation/core/forward_runner.py /app/src/components/backpropagation/core/
COPY src/components/backpropagation/mc_dropout/forward_runner.py /app/src/components/backpropagation/mc_dropout/
COPY src/components/const.py /app/src/components/
COPY src/components/dataset/features/local/local_frequencies_calculator.py /app/src/components/dataset/features/local/
COPY src/components/dataset/features/local/local_recencies_calculator.py /app/src/components/dataset/features/local/
COPY src/components/device/mover.py /app/src/components/device/
COPY src/components/device/selector.py /app/src/components/device/
COPY src/components/inference/autoregressive_rollout/runner.py /app/src/components/inference/autoregressive_rollout/
COPY src/components/logs/levels/debug_logger.py /app/src/components/logs/levels/
COPY src/components/logs/levels/error_logger.py /app/src/components/logs/levels/
COPY src/components/logs/levels/info_logger.py /app/src/components/logs/levels/
COPY src/components/logs/levels/utils/logger.py /app/src/components/logs/levels/utils
COPY src/components/loss/calculator.py /app/src/components/loss/
COPY src/components/model/mode/setter.py /app/src/components/model/mode/
COPY src/components/model/lstm.py /app/src/components/model/
COPY src/components/time/transforms/trig_decoder.py /app/src/components/time/transforms/
COPY src/components/time/transforms/trig_encoder.py /app/src/components/time/transforms/
COPY src/components/yaml/io/loader.py /app/src/components/yaml/io/
COPY src/const.py /app/src/

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Set port
ARG PREDICTOR_SERVICE_PORT
EXPOSE ${PREDICTOR_SERVICE_PORT}

# Move to non-root user
USER appuser

# Run microservice
CMD ["python", "-m", "src.api.services.predictor.predictor_service_grpc_server"]
