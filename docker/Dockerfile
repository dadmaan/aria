# =============================================================================
# ARIA Music Generation - GPU-Enabled Dockerfile
# =============================================================================
# Multi-stage build with CUDA 12.1 + cuDNN 8 for PyTorch GPU acceleration
#
# Build: docker build -t aria-music-generation:gpu .
# Run:   docker run --gpus all -it aria-music-generation:gpu
# =============================================================================

# =============================================================================
# Stage 1: Builder - Install dependencies and compile packages
# =============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel cython

# Install PyTorch with CUDA 12.1 support (separate layer for caching)
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchaudio>=2.0.0 \
    torchvision>=0.15.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Clone and install GHSOM packages
WORKDIR /tmp/ghsom
RUN git clone https://github.com/dadmaan/ghsom-py.git \
    && git clone https://github.com/dadmaan/ghsom-toolkits.git \
    && pip install --no-cache-dir -e ./ghsom-py \
    && pip install --no-cache-dir -e ./ghsom-toolkits

# Copy requirements and install remaining dependencies
COPY requirements.txt /tmp/requirements.txt

# Install dependencies (excluding local editable packages and torch which is already installed)
RUN grep -v "^-e \." /tmp/requirements.txt | \
    grep -v "^torch" | \
    grep -v "^torchaudio" | \
    grep -v "^torchvision" | \
    pip install --no-cache-dir -r /dev/stdin

# Install Jupyter for notebook support
RUN pip install --no-cache-dir jupyterlab ipykernel ipywidgets

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS runtime

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/workspace:${PYTHONPATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy GHSOM packages from builder and reinstall in runtime
COPY --from=builder /tmp/ghsom/ghsom-py /opt/ghsom-py
COPY --from=builder /tmp/ghsom/ghsom-toolkits /opt/ghsom-toolkits
RUN pip install --no-cache-dir -e /opt/ghsom-py -e /opt/ghsom-toolkits

# Create non-root user for security (match host UID/GID when provided)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid ${GROUP_ID} appuser \
    && useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /workspace

# Create necessary directories with proper permissions
RUN mkdir -p /workspace/artifacts \
    /workspace/outputs \
    /workspace/logs \
    /workspace/data \
    /workspace/notebooks \
    && chown -R appuser:appuser /workspace

# Copy project files
COPY --chown=appuser:appuser src/ /workspace/src/
COPY --chown=appuser:appuser scripts/ /workspace/scripts/
COPY --chown=appuser:appuser configs/ /workspace/configs/
COPY --chown=appuser:appuser setup.py pyproject.toml requirements.txt README.md /workspace/

# Install the main package in editable mode
RUN pip install --no-cache-dir -e /workspace

# Switch to non-root user
USER appuser

# Health check - verify PyTorch and CUDA availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import torch; import tianshou; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

# Default command - keep container running for interactive use
# To run training: docker-compose exec music-generation-rl python scripts/training/run_training.py --config configs/training.yaml
CMD ["tail", "-f", "/dev/null"]
