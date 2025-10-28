"""Configuration loading and validation helpers for the RoboDog brain."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import json

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_CONFIG_PATH = Path("vct/config.yaml")


def _clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class BehaviorDefaults(BaseModel):
    """Validated defaults for the behaviour policy."""

    model_config = ConfigDict(extra="ignore")

    energy_level: float = Field(default=0.6, ge=0.0, le=1.0)
    proximity: float = Field(default=0.5, ge=0.0, le=1.0)
    threat_level: float = Field(default=0.1, ge=0.0, le=1.0)
    social_context: float = Field(default=0.5, ge=0.0, le=1.0)
    context: dict[str, float] = Field(default_factory=dict)

    @field_validator("context", mode="after")
    @classmethod
    def clamp_context(cls, value: Mapping[str, Any]) -> dict[str, float]:
        if not value:
            return {}
        return {str(k): _clamp_unit_interval(v) for k, v in value.items()}


class TTSOptions(BaseModel):
    """Settings for the text-to-speech engine factory."""

    model_config = ConfigDict(extra="allow")

    provider: str = Field(default="gtts", min_length=1)
    language: str = Field(default="uk", min_length=1)
    voice: str | None = None
    slow: bool = False


class RoboDogConfig(BaseModel):
    """Top level configuration for :class:`RoboDogBrain`."""

    model_config = ConfigDict(extra="ignore")

    latency_budget_ms: int = Field(default=300, ge=0)
    reward_cooldown_s: float = Field(default=3.0, ge=0.0)
    weights: dict[str, float] = Field(default_factory=dict)
    commands_map: dict[str, str] = Field(default_factory=dict)
    reward_triggers: dict[str, bool] = Field(default_factory=dict)
    mood_initial: str = Field(default="CALM")
    behavior_defaults: BehaviorDefaults = Field(default_factory=BehaviorDefaults)
    tts: TTSOptions = Field(default_factory=TTSOptions)

    @field_validator("weights", mode="after")
    @classmethod
    def validate_weights(cls, weights: Mapping[str, Any]) -> dict[str, float]:
        clean: dict[str, float] = {}
        for key, raw_value in (weights or {}).items():
            value = float(raw_value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Weight '{key}' must be between 0 and 1 inclusive, got {value}")
            clean[str(key)] = value
        return clean

    @field_validator("commands_map", mode="after")
    @classmethod
    def normalise_commands(cls, commands: Mapping[str, Any]) -> dict[str, str]:
        return {str(k).strip(): str(v).strip() for k, v in (commands or {}).items()}

    @field_validator("reward_triggers", mode="after")
    @classmethod
    def ensure_bools(cls, triggers: Mapping[str, Any]) -> dict[str, bool]:
        return {str(k): bool(v) for k, v in (triggers or {}).items()}


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValidationError.from_exception_data(
            "ConfigRootTypeError",
            [
                {
                    "type": "type_error.dict",
                    "loc": ("__root__",),
                    "msg": "Configuration root must be a mapping",
                    "input": data,
                }
            ],
        )
    return data


def _deep_merge(base: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_dotted(overrides: Mapping[str, Any]) -> dict[str, Any]:
    expanded: dict[str, Any] = {}
    for key, value in overrides.items():
        if isinstance(value, Mapping):
            value_dict = _expand_dotted(value)
            expanded[key] = _deep_merge(expanded.get(key, {}), value_dict)
            continue
        current: MutableMapping[str, Any] = expanded
        parts = str(key).split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})  # type: ignore[assignment]
        current[parts[-1]] = value
    return expanded


def parse_override_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def overrides_from_iter(expressions: Iterable[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for expr in expressions:
        if "=" not in expr:
            raise ValueError(f"Invalid override expression '{expr}', expected KEY=VALUE")
        key, raw_value = expr.split("=", 1)
        overrides[key.strip()] = parse_override_value(raw_value.strip())
    return overrides


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RoboDogConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = _read_config_file(config_path)
    if overrides:
        nested = _expand_dotted(overrides)
        raw = _deep_merge(raw, nested)
    return RoboDogConfig.model_validate(raw)

