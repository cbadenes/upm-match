"""Flask application factory for the UPM Match recommendation interface."""

import os

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix


def create_app() -> Flask:
    # The whole stack is served under a single subpath (APP_BASE_PATH, e.g. /upm-match).
    # Nginx preserves that prefix end-to-end (no strip), so the real request path already carries it.
    # We therefore mount the blueprint under the prefix instead of relying on an X-Forwarded-Prefix header.
    # Empty value => local dev (uv run main.py) keeps "/".
    base_path = os.environ.get("APP_BASE_PATH", "").rstrip("/")

    app = Flask(
        __name__,
        template_folder="templates",
        static_url_path=f"{base_path}/static",
    )
    # Honour the reverse-proxy headers: X-Forwarded-Proto so url_for(_external=True) builds https links in production (traffic between host Nginx and the container is plain http) and X-Forwarded-For for the real client IP.
    # The subpath is NOT taken from a header here (it lives in the real path), so x_prefix is intentionally left at 0.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    app.config["APPLICATION_ROOT"] = base_path or "/"

    from .routes import bp  # noqa: PLC0415

    app.register_blueprint(bp, url_prefix=base_path or None)
    return app
