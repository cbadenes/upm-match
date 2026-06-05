import os
import sys
from pathlib import Path

from loguru import logger

_IN_DOCKER = os.environ.get("IN_DOCKER", "false").lower() == "true"

# Airflow captura stderr de cada task y lo muestra en la GUI.
# En local se añade además un fichero rotativo en logs/.


def setup_logging(name: str, level: str | int = "INFO") -> None:
    """Configura Loguru para el script dado.

    En Docker solo usa stderr (Airflow lo captura).
    En local añade también un fichero rotativo en logs/<name>.log.

    :param name: nombre del fichero de log sin extensión (p. ej. "create_pdis")
    :param level: nivel mínimo para stderr (default INFO)
    """
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=not _IN_DOCKER)

    if not _IN_DOCKER:
        log_path = Path("logs") / f"{name}.log"
        log_path.parent.mkdir(exist_ok=True)
        logger.add(log_path, level="DEBUG", rotation="10 MB", retention="30 days")
