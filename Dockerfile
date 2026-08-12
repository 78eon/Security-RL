# RLRedTeam training image — simulation only, no network access at runtime.
FROM python:3.11-slim

WORKDIR /app

# tk is required by nasim's matplotlib-backed renderer; git for provenance stamping.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git tk \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch. The default PyPI wheel pulls ~2.5GB of CUDA libraries that are
# dead weight on this project's compute budget (16GB RAM, no GPU — charter R2).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[dev]"

COPY configs ./configs
COPY data ./data
COPY scripts ./scripts
COPY tools ./tools
COPY tests ./tests

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

CMD ["pytest", "-q"]
