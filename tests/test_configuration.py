from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vct.configuration import (
    load_config,
    overrides_from_iter,
    parse_override_value,
)


def test_load_config_from_yaml():
    cfg = load_config("vct/config.yaml")
    assert cfg.reward_cooldown_s == 3
    assert cfg.commands_map["сидіти"] == "SIT"


def test_load_config_with_overrides(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "reward_cooldown_s": 4,
                "commands_map": {"сидіти": "SIT", "лежати": "LIE_DOWN"},
                "reward_triggers": {"SIT": False, "LIE_DOWN": True},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(
        config_path,
        overrides={"weights.stimulus": 0.75, "reward_triggers.SIT": True},
    )
    assert cfg.weights["stimulus"] == pytest.approx(0.75)
    assert cfg.reward_triggers["SIT"] is True
    assert cfg.reward_triggers["LIE_DOWN"] is True


def test_invalid_config_raises(tmp_path: Path):
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("reward_cooldown_s: -2\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(bad_path)


def test_override_parser_handles_types():
    assert parse_override_value("true") is True
    assert parse_override_value("off") is False
    assert parse_override_value("7") == 7
    assert parse_override_value("0.25") == pytest.approx(0.25)
    overrides = overrides_from_iter(["reward_triggers.SIT=true", "weights.mood=0.2"])
    assert overrides["reward_triggers.SIT"] is True
    assert overrides["weights.mood"] == pytest.approx(0.2)
