# --- Build stage ---------------------------------------------------------
# We install dependencies in a builder layer so the final image stays slim.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# System deps needed to build wheels (e.g. xgboost on slim base images)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Now copy and install the project itself
COPY setup.py ./
COPY src/ ./src/
RUN pip install --no-deps .

# --- Runtime stage -------------------------------------------------------
FROM python:3.11-slim AS runtime

# PYTHONPATH inclusion fixes the ModuleNotFoundError by linking /app/src
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    MLFLOW_TRACKING_URI=file:///app/mlruns

WORKDIR /app

# Copy Python deps + project metadata paths from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the MLflow tracking store and the source tree so MLflow can
# resolve `models:/churn_classifier/Production`
COPY mlruns/ /app/mlruns/
COPY src/ /app/src/
COPY setup.py /app/

# -> WINDOWS PATH REWRITE HACK <-
# Use native Linux tools to find and replace the local Windows paths
RUN find /app/mlruns -type f -name "*.yaml" -exec sed -i 's|file:///.*/mlruns|file:///app/mlruns|g' {} +

EXPOSE 8080

# Serve the Production model. --no-conda is important because the runtime
# image is already configured and there is no conda available.
CMD ["mlflow", "models", "serve", \
     "-m", "models:/churn_classifier/Production", \
     "-h", "0.0.0.0", \
     "-p", "8080", \
     "--no-conda"]