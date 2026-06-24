import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

_config: dict = {}


def load_config(config_path: str = "config/config.yaml", env_file: str = ".env") -> dict:
    """Load config.yaml then overlay any matching environment variables."""
    global _config

    # Load .env file if it exists (silently skip if not present)
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)

    # Read YAML
    with open(config_path, "r") as f:
        _config = yaml.safe_load(f) or {}

    # Overlay key env vars so secrets never live in YAML
    _config.setdefault("env", {})
    _config["env"]["alpaca_api_key"]    = os.getenv("ALPACA_API_KEY", "")
    _config["env"]["alpaca_api_secret"] = os.getenv("ALPACA_API_SECRET", "")
    _config["env"]["environment"]       = os.getenv("ENVIRONMENT", "development")
    _config["env"]["log_level"]         = os.getenv(
        "LOG_LEVEL", _config.get("logging", {}).get("level", "INFO")
    )

    return _config


def get_config() -> dict:
    """Return the already-loaded config. Call load_config() first."""
    if not _config:
        load_config()
    return _config
