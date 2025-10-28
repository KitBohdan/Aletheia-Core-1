
import pytest
from vct.configuration import load_config, RoboDogConfig, _expand_dotted

def test_expand_dotted():
    got = _expand_dotted({"weights.stimulus": 0.25, "tts.provider": "print"})
    assert got["weights"]["stimulus"] == 0.25
    assert got["tts"]["provider"] == "print"

def test_load_config_with_overrides():
    cfg = load_config(overrides={"weights.stimulus": 0.25})
    assert isinstance(cfg, RoboDogConfig)
    assert 0 <= cfg.latency_budget_ms
