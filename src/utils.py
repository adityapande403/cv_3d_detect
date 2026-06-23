from pathlib import Path
import logging
import yaml
from typing import Any, Dict


def load_config(path: Path) -> Dict[str, Any]:
    """Load YAML config from `path` and return as a dict."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure root logger to write to stdout and optional file."""
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)
