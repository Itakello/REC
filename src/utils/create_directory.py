from pathlib import Path

from itakello_logging import ItakelloLogging

logger = ItakelloLogging().get_logger(__name__)


def create_directory(path: Path) -> Path:
    try:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created new directory: {path}")
        else:
            logger.debug(f"Directory already exists: {path}")
        return path
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        raise
