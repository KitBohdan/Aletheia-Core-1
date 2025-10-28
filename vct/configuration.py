"""Configuration loading and validation helpers for the RoboDog brain."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast

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
    tts: TTSOptions = Field(default_factory=TTSOptions)
    behavior_defaults: BehaviorDefaults = Field(default_factory=BehaviorDefaults)

    def action_for_command(self, text: str) -> str:
        normalised = text.strip().lower().replace(" ", "")
        for key, value in self.commands_map.items():
            if key.replace(" ", "") in normalised:
                return value
        return "NONE"


class RewardSchedule(BaseModel):
    """Simple mapping of actions to reward availability."""

    model_config = ConfigDict(extra="ignore")

    triggers: dict[str, bool] = Field(default_factory=dict)

    @field_validator("triggers", mode="after")
    @classmethod
    def _coerce_bool(cls, value: Mapping[str, Any]) -> dict[str, bool]:
        return {str(k): bool(v) for k, v in value.items()}


def parse_override_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "on", "yes"}:
        return True
    if lowered in {"false", "off", "no"}:
        return False
    try:
        if "." in lowered:
            return float(lowered)
        return int(lowered)
    except ValueError:
        return value


def overrides_from_iter(items: Iterable[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item!r}")
        key, raw_value = item.split("=", 1)
        overrides[key.strip()] = parse_override_value(raw_value)
    return overrides


def _deep_merge_dict(
    base: Mapping[str, Any], updates: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge two mapping objects."""

    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dict(existing, value)
        else:
            merged[key] = value
    return merged


def _apply_overrides(
    data: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply dotted overrides to a configuration mapping."""

    merged: dict[str, Any] = dict(data)
    for raw_key, value in overrides.items():
        key = raw_key.strip()
        if not key:
            continue
        if "." not in key:
            existing = merged.get(key)
            if isinstance(existing, Mapping) and isinstance(value, Mapping):
                merged[key] = _deep_merge_dict(existing, value)
            else:
                merged[key] = value
            continue

        parts = [part for part in key.split(".") if part]
        if not parts:
            continue

        cursor: dict[str, Any] = merged
        for part in parts[:-1]:
            existing = cursor.get(part)
            if isinstance(existing, Mapping):
                next_node: dict[str, Any] = dict(existing)
            else:
                next_node = {}
            cursor[part] = next_node
            cursor = next_node

        final_key = parts[-1]
        existing = cursor.get(final_key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            cursor[final_key] = _deep_merge_dict(existing, value)
        else:
            cursor[final_key] = value

    return merged


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> RoboDogConfig:
    if path is None:
        path = DEFAULT_CONFIG_PATH
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {raw_path}")

    if raw_path.suffix.lower() in {".yaml", ".yml"}:
        with raw_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    elif raw_path.suffix.lower() == ".json":
        data = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported configuration format: {raw_path.suffix}")

    if overrides:
        data = _apply_overrides(data, overrides)

    try:
        return cast(RoboDogConfig, RoboDogConfig.model_validate(data))
    except ValidationError as exc:  # pragma: no cover - validation errors bubble up
        raise ValueError(f"Invalid configuration: {exc}") from exc
