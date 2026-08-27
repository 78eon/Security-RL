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

# CP-04: a dedicated unprivileged user. Training never needs root, and running
# as root means a bug in the environment writes with root authority on every
# bind mount.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin rlredteam \
    && mkdir -p /app/runs /app/results \
    && chown -R 10001:10001 /app/runs /app/results \
    # /app is bind-mounted from the host and so is owned by a different uid than
    # the one this image runs as. Git refuses to read a repository it considers
    # foreign ("dubious ownership"), which would silently strip commit
    # provenance from every run. The mount is read-only, so trusting it here is
    # safe: nothing in the container can write to history.
    && git config --system --add safe.directory /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0

USER 10001:10001

CMD ["pytest", "-q"]
