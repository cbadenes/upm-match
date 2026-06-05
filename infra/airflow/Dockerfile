FROM apache/airflow:2.10.4-python3.12

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

USER airflow
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/airflow/.local/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT=/home/airflow/upm_venv
RUN uv python install 3.13

# Pre-install project dependencies (baked into image, no volume permissions needed)
# Build context is the project root, so pyproject.toml and uv.lock are accessible.
COPY --chown=airflow:root pyproject.toml uv.lock /tmp/upm_deps/
RUN cd /tmp/upm_deps && uv sync --frozen --no-install-project
