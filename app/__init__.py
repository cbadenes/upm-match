"""Flask application factory for the UPM Match recommendation interface."""

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")
    # Reads X-Forwarded-Prefix set by Traefik so url_for() generates correct
    # paths when the app is served under a subpath (e.g. /app).
    app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1)

    from .routes import bp  # noqa: PLC0415

    app.register_blueprint(bp)
    return app
