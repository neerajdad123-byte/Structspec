FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for llama-cpp-python builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first for layer caching
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package
RUN pip install --no-cache-dir -e ".[all]"

# Default entrypoint
ENTRYPOINT ["structspec-qwen"]
CMD ["--help"]
