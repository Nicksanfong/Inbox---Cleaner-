import logging
import logging.handlers
from pathlib import Path
from config.loader import get_config


def setup_logging() -> logging.Logger:
    """Configure root logger to write to both console and a rotating file."""
    cfg = get_config().get("logging", {})
    level_name = get_config().get("env", {}).get("log_level", cfg.get("level", "INFO"))
    level = getattr(logging, level_name.upper(), logging.INFO)

    log_file = Path(cfg.get("file", "logs/trading.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)

    # Rotating file handler (10 MB × 5 backups)
    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=cfg.get("max_bytes", 10_485_760),
        backupCount=cfg.get("backup_count", 5),
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)

    root.addHandler(ch)
    root.addHandler(fh)

    return root
