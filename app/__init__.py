"""Flask application factory for the UPM Match recommendation interface."""

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")

    from .routes import bp  # noqa: PLC0415

    app.register_blueprint(bp)
    return app
