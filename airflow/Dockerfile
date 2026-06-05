FROM apache/airflow:2.10.4-python3.12

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

USER airflow
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/airflow/.local/bin:$PATH"
# Pre-fetch Python 3.13 so first DAG run doesn't need internet access
RUN uv python install 3.13
