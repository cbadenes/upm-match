FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first (layer is cached until pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy application code (build context is the project root)
COPY app/       app/
COPY dbs/       dbs/
COPY models/    models/
COPY utils/     utils/

ENV PYTHONUNBUFFERED=1

# Gunicorn listens on the app's fixed default port (5000). The Nginx upstream
# (app:${APP_PORT}) must point at this same port — keep APP_PORT in sync.
# --timeout is generous because each request makes synchronous LLM/embedding
# calls to a shared Ollama instance; a cold model load can take minutes.
CMD ["uv", "run", "gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "300", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "app:create_app()"]
