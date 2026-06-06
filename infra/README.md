# UPM Match — Local Infrastructure

Docker configuration for the local development environment.
Contains everything needed to run the full stack: databases, LLM, object storage, reverse proxy and pipeline orchestrator.

---

## Structure

```
infra/
├── docker-compose.yaml    # All service definitions
├── app/
│   └── Dockerfile         # Flask app image (python:3.13-slim + uv + gunicorn)
├── airflow/
│   ├── Dockerfile         # Airflow image extended with uv + Python 3.13
│   └── dags/
│       └── upm_match.py   # Data pipeline DAG (6 phases)
├── traefik/
│   └── dynamic.yml        # Traefik routes and middlewares (file provider)
└── README.md              # This file
```

---

## Prerequisites

- **Docker Desktop** ≥ 4.x with WSL2 enabled
- In Docker Desktop $\rightarrow$ Settings $\rightarrow$ General: enable **"Expose daemon on tcp://localhost:2375 without TLS"**
- A `.env` file at the project root (see Environment variables below)

---

## Environment variables

Copy `.env.sample` (project root) to `.env` and fill in the real values:

```bash
cp .env.sample .env
```

The `.env.sample` file documents every variable along with the commands needed to generate the Airflow keys. The actual `.env` is never committed.

---

## Setup

All commands must be run from the **project root** using `--env-file .env` so that Docker Compose finds the `.env` file (it looks in the compose file's directory by default, which is `infra/`, not the project root).

### First time

```bash
# 1. Build the Airflow image and initialise its database
docker compose --env-file .env -f infra/docker-compose.yaml up airflow-init --build

# 2. Start all services
docker compose --env-file .env -f infra/docker-compose.yaml up -d

# 3. Pull Ollama models (only needed once)
docker exec tfm_ollama ollama pull qwen2.5:7b-instruct
docker exec tfm_ollama ollama pull nomic-embed-text-v2-moe
```

### Daily usage

```bash
# Start
docker compose --env-file .env -f infra/docker-compose.yaml up -d

# Stop (data is preserved in volumes)
docker compose --env-file .env -f infra/docker-compose.yaml down

# Check status
docker compose --env-file .env -f infra/docker-compose.yaml ps

# Tail logs for a specific service
docker compose --env-file .env -f infra/docker-compose.yaml logs -f airflow-webserver

# Tail app logs (includes both Gunicorn access log and loguru output)
docker compose --env-file .env -f infra/docker-compose.yaml logs -f app
```

### Rebuilding the app after code changes

The app image bakes the source code at build time, so any change to `app/`, `dbs/`, `models/` or `utils/` requires a rebuild:

```bash
docker compose --env-file .env -f infra/docker-compose.yaml up -d --build app
```

For rapid iteration, `uv run main.py` (Flask dev server on port 5000) still works locally without Docker as long as Qdrant and Ollama ports are exposed in `docker-compose.yaml` (the commented-out `ports:` blocks).

---

## Service access

| Service | URL | Credentials |
|---|---|---|
| **Traefik Dashboard** | http://traefik.localhost/dashboard/ | No login |
| **UPM Match App** | http://localhost/app | No login |
| **Airflow** | http://localhost/airflow | `admin` / `AIRFLOW_ADMIN_PASSWORD` |
| **Memgraph Lab** | http://lab.localhost | `MEMGRAPH_USERNAME` / `MEMGRAPH_PASSWORD` |
| **MinIO Console** | http://minio.localhost | `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` |
| **MinIO S3 API** | http://localhost/s3 | AWS-style credentials (`MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`) |
| **Ollama API** | http://localhost/ollama | No login |
| **Qdrant** | http://qdrant.localhost/dashboard | Include the API key to see the GUI |


> `*.localhost` subdomains resolve automatically in modern browsers.
> If a subdomain does not load, add it to `C:\Windows\System32\drivers\etc\hosts`:
> ```
> 127.0.0.1  qdrant.localhost lab.localhost minio.localhost traefik.localhost
> ```

---

## Data pipeline (Airflow)

The DAG `upm_match_pipeline` in `airflow/dags/upm_match.py` runs the 6 phases sequentially, with phases 05 and 06 running in parallel at the end:

```
01_find_ids -> 02_get_pdis_data -> 03_create_pdis -> 04_enrich_profile -> +-> 05_create_keywords
                                                                          +-> 06_create_summaries
```

To trigger it: http://localhost/airflow $\rightarrow$ DAG `upm_match_pipeline` $\rightarrow$ ▶ button.

Dependencies are pre-installed in the Docker image, so the first task run starts immediately.

---

## Architecture decisions

### Why Traefik as the reverse proxy

The target is to migrate this infrastructure to K8s at some point. Traefik was chosen because:

1. **Native dynamic discovery in K8s**: Traefik reads `Ingress` and `IngressRoute` resources automatically without per-service configuration. Nginx requires manual reloads or a separate controller.

2. **Same mental model in Docker and K8s**: locally, routes are declared in `traefik/dynamic.yml`; in production they become `IngressRoute` CRDs. The transition is direct.

3. **Declarative middlewares**: authentication, IP filtering, rate limiting and path stripping are declared alongside the routes in one file, with no `location` blocks or extra modules.

4. **Built-in dashboard**: instant visibility of active routes without external tooling.

**Note on Docker Desktop for Windows**: Traefik v3 uses the Docker Go SDK, which negotiates the API version starting from v1.24. 
Docker Desktop 29.x requires a minimum of v1.40 and rejects that with HTTP 400. 
For this reason the **file provider** is used instead of the Docker provider: Traefik reads routes from `traefik/dynamic.yml` without needing Docker socket access. 
This limitation does not exist on Linux or K8s.

---

### Why subdomains for GUIs and subpaths for APIs

The split is a technical consequence of how modern web applications are built, not a stylistic choice.

**Why subpaths break SPAs:**

A SPA (Single Page Application) such as the Qdrant UI or MinIO Console is compiled with webpack/Vite. During the build, all asset paths (JS, CSS, fonts) are written with an absolute prefix. 
The default is `/`, producing HTML like: 

```html
<script src="/assets/main-3f2a1b.js"></script>
```

When Traefik routes `localhost/qdrant` with `StripPrefix /qdrant`, Qdrant receives the request at `/`, serves the HTML and the browser tries to load `localhost/assets/main-3f2a1b.js`. 
Traefik has no route for `/assets/...` $\rightarrow$ 404 $\rightarrow$ the page loads blank.

The only clean fix is for the application to expose a `BASE_PATH` environment variable so the build compiles assets with that prefix. None of the SPAs in this stack support it.

**Why subpaths work for some services:**

| Service | Type | Why subpaths work |
|---|---|---|
| UPM Match App | Flask app | `ProxyFix(x_prefix=1)` reads `X-Forwarded-Prefix: /app` sent by Traefik; `url_for()` generates correct paths; no compiled assets |
| Airflow | Flask app | Respects `AIRFLOW__WEBSERVER__BASE_URL`: generates all URLs with the configured prefix, no StripPrefix needed |
| Ollama | Pure REST API | Serves only JSON, no HTML assets $\rightarrow$ StripPrefix works |
| MinIO S3 API | Pure REST API | Same as above |

**Why subdomains work for all services:**

With `Host('qdrant.localhost')`, Traefik forwards traffic without modifying the path. 
Qdrant receives requests at `/`, serves HTML with `<script src="/assets/main.js">` and the browser requests `qdrant.localhost/assets/main.js`. 
Traefik routes that to Qdrant too (same `Host` rule) $\rightarrow$ 200. 
No conflicts.

In Kubernetes this maps directly to Ingress `spec.rules[].host` per service.

---

## Security middlewares

Middlewares are defined in `traefik/dynamic.yml` but disabled. To activate them:

**1. IP AllowList** (for APIs such as Qdrant or Ollama):

Uncomment the definition in `traefik/dynamic.yml`:
```yaml
    api-allowlist:
      ipAllowList:
        sourceRange:
          - "127.0.0.1/32"
          - "192.168.1.0/24"
```

Then add it to the target router:
```yaml
    qdrant:
      ...
      middlewares: [api-allowlist]
```

**2. Basic Auth** (for GUIs such as Memgraph Lab):

Generate the hash:
```bash
docker run --rm httpd:2.4-alpine htpasswd -nbB admin YOUR_PASSWORD
# Output: admin:$2y$05$HASH_GENERATED
```

Uncomment and replace the hash in `traefik/dynamic.yml` (no `$` escaping here —
this file is not processed by Docker Compose):
```yaml
    gui-auth:
      basicAuth:
        users:
          - "admin:$2y$05$HASH_GENERATED"
```

Then add it to the target router:
```yaml
    lab:
      ...
      middlewares: [gui-auth]
```

After any change to `traefik/dynamic.yml`, restart Traefik to reload it:
```bash
docker compose --env-file .env -f infra/docker-compose.yaml restart traefik
```
