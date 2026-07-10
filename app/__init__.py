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

    # Secret used to sign the session cookie that carries the evaluation flow (consent + selected PDI).
    # Must be STABLE and SHARED across gunicorn workers, otherwise a cookie signed by one worker is rejected by another.
    # Prefer a dedicated APP_SECRET_KEY; fall back to the already-present AIRFLOW_SECRET_KEY so it works today without new env, and to a dev key as a last resort.
    secret = os.environ.get("APP_SECRET_KEY") or os.environ.get("AIRFLOW_SECRET_KEY")
    if not secret:
        secret = "dev-insecure-change-me"
        import warnings  # noqa: PLC0415

        warnings.warn(
            "No APP_SECRET_KEY/AIRFLOW_SECRET_KEY set; using an insecure dev key "
            "for session signing. Set APP_SECRET_KEY in production.",
            stacklevel=2,
        )
    app.secret_key = secret
    # Session cookie hardening (survey has no cross-site needs).
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )

    from .evaluation import eval_bp  # noqa: PLC0415
    from .routes import bp  # noqa: PLC0415

    app.register_blueprint(bp, url_prefix=base_path or None)
    app.register_blueprint(eval_bp, url_prefix=base_path or None)

    # Branded error pages instead of the raw Werkzeug defaults.
    from flask import render_template  # noqa: PLC0415

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("error.html", code=404), 404

    @app.errorhandler(500)
    def server_error(_e):
        return render_template("error.html", code=500), 500

    return app
