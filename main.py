import logging
from config.loader import load_config
from config.logger import setup_logging


def main():
    load_config()
    setup_logging()
    log = logging.getLogger(__name__)

    log.info("=" * 50)
    log.info("Algorithmic Trading Platform")
    log.info("=" * 50)

    from config.loader import get_config
    cfg = get_config()
    env = cfg.get("env", {})
    trading = cfg.get("trading", {})

    log.info("Environment : %s", env.get("environment", "development"))
    log.info("Trading mode: %s", trading.get("mode", "paper"))
    log.info("Symbols     : %s", ", ".join(trading.get("symbols", [])))
    log.info("=" * 50)
    log.info("System ready")


if __name__ == "__main__":
    main()
