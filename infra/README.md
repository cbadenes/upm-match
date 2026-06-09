# UPM Match — Infrastructure

Docker configuration for the full stack: databases, LLM, object storage, reverse proxy and pipeline orchestrator.
Everything is served under a **single subpath**(`/upm-match`) behind one host port, so the production server needs no subdomains.

---

## Routing model (subpaths, not subdomains)

The production server already runs its own host-managed Nginx and **cannot create subdomains**: everything must hang off one URL, `https://<domain>/upm-match/`.
The host Nginx forwards that subpath to a single local port.
Inside the stack a container Nginx fans the request out to each service:

```
Browser ──https──► Host Nginx ──http──► 127.0.0.1:HOST_PORT ──► container Nginx :80 ──► service
            (TLS, public)      (loopback, private port)        (infra/nginx/...)      (app/airflow/lab/...)
```

The prefix `/upm-match` is **preserved end-to-end** (never stripped at the host).
Each app therefore sees the exact path the browser sees and generates correct links.
The host Nginx is **not ours to edit** — we only hand the admin the block in [`nginx/host-nginx.example.conf`](nginx/host-nginx.example.conf).

---

## Structure

```
infra/
├── docker-compose.yaml          # All service definitions
├── app/
│   └── Dockerfile               # Flask app image (python:3.13-slim + uv + gunicorn)
├── airflow/
│   ├── Dockerfile               # Airflow image extended with uv + Python 3.13
│   └── dags/upm_match.py        # Data pipeline DAG (6 phases)
├── nginx/
│   ├── default.conf.template    # Container Nginx routing (rendered by envsubst)
│   └── host-nginx.example.conf  # Block to hand to the host-Nginx admin
├── traefik/
│   └── dynamic.yml.bak          # Previous Traefik config (kept for reference only)
└── README.md                    # This file
```

---

## Prerequisites

- **Docker Desktop** $\geq 4.x$ with WSL2 enabled (NVIDIA GPU passthrough for Ollama).
- A `.env` file at the project root (see below).

---

## Environment variables

Copy `.env.sample` (project root) to `.env` and fill in the real values:

```bash
cp .env.sample .env
```

The variables that drive the routing:

| Variable | Example (local) | Purpose |
|---|---|---|
| `APP_BASE_PATH` | `/upm-match` | The subpath everything hangs under. Used by the Nginx template, the Flask blueprint and the Lab `BASE_PATH`. |
| `HOST_PORT` | `1234` | Loopback port the host Nginx forwards to. **Keep private**: published only on `127.0.0.1`. |
| `PUBLIC_BASE_URL` | `http://localhost:1234/upm-match` | Full external base URL (scheme + host + prefix). Drives Airflow's `BASE_URL`. In production: `https://<domain>/upm-match`. |

`.env` is never committed (it is in `.gitignore`).
`APP_BASE_PATH` must be the path component of `PUBLIC_BASE_URL`.

---

## Setup

All commands run from the **project root** with `--env-file .env` (Compose looks for `.env` in the compose file's directory, `infra/`, by default, not the root).

### First time

```bash
# 1. Build the Airflow image and initialise its metadata DB
docker compose --env-file .env -f infra/docker-compose.yaml up airflow-init --build

# 2. Build and start everything
docker compose --env-file .env -f infra/docker-compose.yaml up -d --build

# 3. Pull the Ollama models (only once)
docker exec -it upm_match_ollama ollama pull qwen2.5:7b-instruct
docker exec -it upm_match_ollama ollama pull nomic-embed-text-v2-moe
docker exec -it upm_match_ollama ollama pull <other_ollama_model>  # You must explicit it in .env to use it
```

### Daily usage

```bash
# Start
docker compose --env-file .env -f infra/docker-compose.yaml up -d

# Stop (data preserved in the upm_match_* volumes)
docker compose --env-file .env -f infra/docker-compose.yaml down

# Status / logs
docker compose --env-file .env -f infra/docker-compose.yaml ps
docker compose --env-file .env -f infra/docker-compose.yaml logs -f nginx
docker compose --env-file .env -f infra/docker-compose.yaml logs -f app

# Rebuild the app after code changes (source is baked at build time)
docker compose --env-file .env -f infra/docker-compose.yaml up -d --build app
```

For rapid iteration, `uv run main.py` still runs the Flask dev server locally without Docker (with `APP_BASE_PATH` unset it serves at `/`).
It needs Qdrant and Ollama reachable (uncomment their `ports:` blocks in `docker-compose.yaml`).

---

## Service access

All public URLs share the same host/port and prefix. Locally that is `http://localhost:1234/upm-match`; in production `https://<domain>/upm-match`.

| Service | URL | Credentials |
|---|---|---|
| **UPM Match App** | `…/upm-match/` | No login |
| **Airflow** | `…/upm-match/airflow` | `admin` / `AIRFLOW_ADMIN_PASSWORD` |
| **Memgraph Lab** | `…/upm-match/lab` | `MEMGRAPH_USERNAME` / `MEMGRAPH_PASSWORD` |
| **Qdrant dashboard** | `…/upm-match/qdrant/dashboard` | Enter `QDRANT_API_KEY` in the UI (top-right key icon) |

Internal-only (no public route, reachable by service name on the Docker network):

| Service | Internal address | Used by |
|---|---|---|
| MinIO (S3) | `minio:9000` | app, all phases |
| Ollama | `ollama:11434` | app, phase 06 |
| Memgraph (Bolt) | `memgraph:7687` | Lab, phases 03/04 |
| PostgreSQL | `postgres:5432` | Airflow |

> For direct host access during local debugging, uncomment the `127.0.0.1:…` `ports:` blocks of `minio`, `memgraph`, `ollama` or `qdrant` in `docker-compose.yaml`.
> They are commented so nothing but Nginx is exposed in production.

---

## Data pipeline (Airflow)

The DAG `upm_match_pipeline` runs the 6 phases, with 05 and 06 in parallel:

```
01_find_ids => 02_get_pdis_data => 03_create_pdis => 04_enrich_profile => +=> 05_create_keywords
                                                                          +=> 06_create_summaries
```

Trigger it at `…/upm-match/airflow` $\Rightarrow$ DAG `upm_match_pipeline` $\Rightarrow$ ▶.

---

## How each service is served under the subpath

The hard part of subpath routing is that single-page apps compile asset paths with an absolute prefix at build time.
If you strip the prefix, the browser then asks for `/<domain>/assets/...` and gets a 404.
The clean fix is for each app to be told its base path.
Here is how each one is handled:

| Service | Strip prefix? | How it becomes base-path aware |
|---|---|---|
| **Flask app** | No | Blueprint mounted at `url_prefix=$APP_BASE_PATH`. `ProxyFix(x_proto)` for https. `url_for()` emits `/upm-match/...`. No compiled assets. |
| **Airflow** | No | `AIRFLOW__WEBSERVER__BASE_URL=$PUBLIC_BASE_URL/airflow` + `ENABLE_PROXY_FIX`. Serves every URL/asset under the prefix natively. |
| **Memgraph Lab** | No | `BASE_PATH=$APP_BASE_PATH/lab` env. v3.10.0 **accepts the multi-segment path** and emits `<base href="/upm-match/lab/">`. WebSocket headers forwarded for live logs. |
| **Qdrant dashboard** | Yes | No base-path support. Served via strip + `sub_filter` + a small injected shim. See below. |

Longest-prefix wins in Nginx, so `…/airflow`, `…/lab` and `…/qdrant/` match before the app's catch-all `…/upm-match/`.

### Qdrant dashboard (the stubborn one)

Qdrant's SPA assumes it lives at the domain root, in two independent ways, so it needs three coordinated tricks (all in the `…/qdrant/` location):

1. **Strip** `…/qdrant` $\Rightarrow$ Qdrant root, so the REST API is reachable at `…/qdrant/<endpoint>` and the SPA at `…/qdrant/dashboard`.

2. **Static assets** are emitted as absolute `/dashboard/...` (HTML `<script>`/`<link>`, web workers, module preloads, `openapi.json`, and CSS `@font-face`).
A `sub_filter` rewrites every `/dashboard/` $\Rightarrow$ `…/qdrant/dashboard/` across `text/html`, `text/javascript`, `application/json` and `text/css` (upstream compression is disabled with `Accept-Encoding ""` so `sub_filter` can see the body).
Only `/dashboard/` (with trailing slash) is rewritten and never the bare `/dashboard`, which would corrupt the app's `…replace(/dashboard$/…)` logic.

3. **REST calls** are built at runtime from `origin + port`, **discarding the path**, so the client hits `/<endpoint>` at the root, which in production never reaches us.
A tiny shim injected into `<head>` (before the bundle) patches `fetch`/`XMLHttpRequest` to re-prefix same-origin requests not already under `…/upm-match/` with `…/qdrant`.
Cross-origin calls are left untouched.

Sections use **hash routing** (`…/dashboard#/collections`), so the URL path stays `…/qdrant/dashboard` and refresh/deep-links never 404.
Enter `QDRANT_API_KEY` via the key icon (top-right) to authenticate.
One harmless leftover: a Qdrant-Cloud-only `cloud/data.json` request 401s in the console — that endpoint does not exist in the self-hosted build.

### Why preserve the prefix (no strip)

Stripping would force every app to reconstruct its public path from an `X-Forwarded-Prefix` header (which not all of them honour).
Preserving it means each app sees the real path and just needs to know its own base, simpler and more robust.
The container Nginx therefore uses `proxy_pass` **without a path**.

### Two robustness details in the container Nginx

- **`absolute_redirect off`**: the container only knows its internal `:80`, not the public host/port.
Relative `Location` headers let the browser keep the original scheme/host/port (otherwise local redirects would drop `:1234`).
- **`proxy_set_header Host $http_host`** and a `map` that prefers the incoming **`X-Forwarded-Proto`** over `$scheme`: between host and container the traffic is plain http, so we propagate the public scheme the host Nginx sent (https) and the exact Host (with port locally), falling back to `$scheme`/`$host` for direct local access.

---

## Security middlewares (disabled)

Both are pre-written and commented in `nginx/default.conf.template`:

1. **IP allow-list**: `allow … / deny all;` to restrict who reaches the stack.
2. **Basic Auth**: generate `htpasswd -B -c infra/nginx/.htpasswd admin`, mount it and uncomment `auth_basic` (server-wide or per `location`).

They are intentionally left off for now (to be decided here or upstream later).

After editing the template, recreate Nginx:

```bash
docker compose --env-file .env -f infra/docker-compose.yaml up -d --force-recreate nginx
```

---

## Pending / fragile

- **MinIO console** has no public route (commented in the template).
It is another root-assuming SPA, and the console image (2025-09-07) ships a trimmed console (object browser only).
If needed, set `MINIO_BROWSER_REDIRECT_URL=$PUBLIC_BASE_URL/console` and enable the `…/console/` location. Expect to iterate on strip VS. no-strip for its assets.
Not required for the pipeline or the app.
- The **Qdrant dashboard** shim relies on the dashboard building REST URLs from the page origin.
If a future Qdrant version changes that, re-check the injected `<head>` shim against the network tab (it must keep API calls under `…/qdrant/`).
