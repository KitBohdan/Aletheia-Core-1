from vct.configuration import load_config, RoboDogConfig, DEFAULT_CONFIG_PATH

def test_load_config_default_file():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(cfg, RoboDogConfig)
    assert cfg.latency_budget_ms >= 0
