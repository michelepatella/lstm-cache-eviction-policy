FROM python:3.12-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash locustuser

# Set working directory
WORKDIR /app/load_test

# Copy requirements and install dependencies
COPY load_test/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p \
    /app/load_test \
    /app/src/ \
    /app/src/components/assertions \
    /app/src/components/dataset/io \
    /app/src/components/logs/levels \
    /app/src/components/logs/levels/utils \
    /app/src/components/yaml/io \
    /app/src/pipeline/config/pydantic/sections \
    /app/src/pipeline/config \
    /app/src/components/dataset/rows/extractions \
    /app/src/components/time/transforms

# Copy all needed files
COPY data/processed/ /app/data/processed/
COPY load_test/locustfile.py /app/load_test/
COPY load_test/helpers.py /app/load_test/
COPY src/components/assertions/choice_field_assertor.py /app/src/components/assertions/
COPY src/components/assertions/min_max_assertor.py /app/src/components/assertions/
COPY src/components/const.py /app/src/components/
COPY src/components/dataset/io/loader.py /app/src/components/dataset/io/
COPY src/components/dataset/io/locator.py /app/src/components/dataset/io/
COPY src/components/dataset/rows/extractions/lasts_extractor.py /app/src/components/dataset/rows/extractions/
COPY src/components/dataset/rows/extractions/sliding_window_extractor.py /app/src/components/dataset/rows/extractions/
COPY src/components/dataset/rows/extractions/tuple_extractor.py /app/src/components/dataset/rows/extractions/
COPY src/components/logs/levels/debug_logger.py /app/src/components/logs/levels/
COPY src/components/logs/levels/error_logger.py /app/src/components/logs/levels/
COPY src/components/logs/levels/utils/logger.py /app/src/components/logs/levels/utils/
COPY src/components/yaml/io/loader.py /app/src/components/yaml/io/
COPY src/components/time/transforms/trig_decoder.py /app/src/components/time/transforms/
COPY src/pipeline/config/pipeline_config.yaml /app/src/pipeline/config/
COPY src/pipeline/config/configurator.py /app/src/pipeline/config/
COPY src/pipeline/config/pydantic/pipeline_config.py /app/src/pipeline/config/pydantic/
COPY src/pipeline/config/pydantic/sections/data_loader_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/config/pydantic/sections/data_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/config/pydantic/sections/dataset_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/config/pydantic/sections/early_stopping_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/config/pydantic/sections/evaluation_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/config/pydantic/sections/logs_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/config/pydantic/sections/loss_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/config/pydantic/sections/model_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/config/pydantic/sections/optimizer_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/config/pydantic/sections/resources_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/config/pydantic/sections/seed_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/config/pydantic/sections/simulations_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/config/pydantic/sections/training_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/config/pydantic/sections/validation_pipeline_config.py /app/src/pipeline/config/pydantic/sections/
COPY src/pipeline/const.py /app/src/pipeline/
COPY src/const.py /app/src/

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Expose Locust web UI port
ARG LOCUST_PORT
EXPOSE ${LOCUST_PORT}

# Switch to non-root user
USER locustuser

# Run Locust
CMD ["sh", "-c", "locust --host ${TARGET_HOST} --web-port ${LOCUST_PORT}"]
