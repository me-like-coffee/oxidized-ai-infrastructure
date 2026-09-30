# ---------------------------------------------------------------------
# STAGE 1: The Development & Automated Test Environment (Option B)
# ---------------------------------------------------------------------
FROM rust:1.85-slim-bookworm AS development

# Install Python 3, pip, and required system build libraries
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Setup isolated Python virtual environment
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Pre-install core open-source machine learning and simulation frameworks
COPY reinforcement_learning/python/requirements.txt ./reinforcement_learning/python/
RUN pip install --no-cache-dir -r ./reinforcement_learning/python/requirements.txt

# Tell PyO3 exactly which Python interpreter binary to bind against during compilation
ENV PYO3_PYTHON="/app/venv/bin/python"

# Copy entire workspace metadata and code layers
COPY Cargo.toml ./
COPY reinforcement_learning/ ./reinforcement_learning/

# ---------------------------------------------------------------------
# STAGE 2: Production Compilation Cache
# ---------------------------------------------------------------------
FROM development AS builder
RUN cargo build --release --bin reinforcement_learning_server

# ---------------------------------------------------------------------
# STAGE 3: Microscopic Production Inference Container (Option A)
# ---------------------------------------------------------------------
FROM debian:bookworm-slim AS production
WORKDIR /app

# Copy the high-performance binary from the builder layer (zero Python dependencies)
COPY --from=builder /app/target/release/reinforcement_learning_server /app/inference_server

EXPOSE 8080
CMD ["./inference_server"]

