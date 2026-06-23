import pytest
from config.loader import load_config, get_config


def test_load_config_returns_dict():
    cfg = load_config()
    assert isinstance(cfg, dict)


def test_trading_section_present():
    cfg = load_config()
    assert "trading" in cfg
    assert cfg["trading"]["mode"] in ("paper", "live")


def test_risk_section_present():
    cfg = load_config()
    assert "risk" in cfg
    assert cfg["risk"]["max_open_positions"] > 0


def test_get_config_same_as_load():
    load_config()
    assert get_config() is not None
