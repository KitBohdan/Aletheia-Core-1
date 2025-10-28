"""Core RoboDog brain logic bound to the behavioural policy."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..behavior.policy import BehaviorInputs, BehaviorPolicy
from ..configuration import DEFAULT_CONFIG_PATH, RoboDogConfig, load_config
from ..engines.stt import WhisperSTT
from ..engines.tts import PrintTTS, create_tts_engine
from ..ethics.guard import EthicsGuard
from ..hardware.gpio_reward import GPIOActuator, SimulatedActuator
from ..utils.logging import get_logger

log = get_logger("RoboDogBrain")


class RoboDogBrain:
    """High-level orchestrator translating commands into actions."""

    def __init__(
        self,
        cfg_path: str | Path | RoboDogConfig | None = None,
        gpio_pin: int | None = None,
        simulate: bool = False,
        *,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(cfg_path, RoboDogConfig):
            self.config = cfg_path
        else:
            path = cfg_path or DEFAULT_CONFIG_PATH
            self.config = load_config(path, overrides=config_overrides)

        # Maintain the legacy public attribute exposed as a mapping for
        # compatibility with earlier integrations.
        self.cfg = self.config.model_dump()

        self.stt = WhisperSTT()
        if simulate:
            self.tts = PrintTTS()
        else:
            self.tts = create_tts_engine(self.config.tts.model_dump())

        self.policy = BehaviorPolicy(self.config.weights)
        self.reward_map: dict[str, bool] = dict(self.config.reward_triggers)
        self.cooldown_s = float(self.config.reward_cooldown_s)
        self.simulate = simulate
        self.actuator = SimulatedActuator() if (simulate or gpio_pin is None) else GPIOActuator(gpio_pin)
        self.guard = EthicsGuard()

        defaults = self.config.behavior_defaults
        self.behavior_defaults = {
            "energy_level": float(defaults.energy_level),
            "proximity": float(defaults.proximity),
            "threat_level": float(defaults.threat_level),
            "social_context": float(defaults.social_context),
        }
        self.behavior_context = {k: float(v) for k, v in defaults.context.items()}

    def _action_from_text(self, text: str) -> str:
        mapping = self.config.commands_map
        normalised = text.strip().lower().replace(" ", "")
        for key, value in mapping.items():
            if key.replace(" ", "") in normalised:
                return value
        return "NONE"

    def _maybe_reward(self, action: str, score: float) -> bool:
        if not self.reward_map.get(action, False):
            return False
        now = time.time()
        if not self.guard.can_reward(now, action, score, self.cooldown_s):
            return False
        self.actuator.trigger(0.4)
        self.guard.note_reward(now)
        return True

    def handle_command(
        self,
        text: str,
        confidence: float = 0.85,
        reward_bias: float = 0.5,
        mood: float = 0.0,
    ) -> dict[str, Any]:
        action = self._action_from_text(text)
        context = dict(self.behavior_context)
        context["action_known"] = 1.0 if action != "NONE" else 0.0
        context["reward_available"] = 1.0 if self.reward_map.get(action, False) else 0.0
        inputs = BehaviorInputs(
            stimulus=1.0 if action != "NONE" else 0.0,
            confidence=confidence,
            reward_bias=reward_bias,
            mood=mood,
            energy_level=self.behavior_defaults["energy_level"],
            proximity=self.behavior_defaults["proximity"],
            threat_level=self.behavior_defaults["threat_level"],
            social_context=self.behavior_defaults["social_context"],
            context=context,
        )
        vector = self.policy.decide(action, inputs)
        rewarded = self._maybe_reward(vector.action, vector.score)
        feedback = f"Дія: {vector.action} score={vector.score:.2f}" + (" — ✅ винагорода" if rewarded else "")
        self.tts.speak(feedback)
        log.info(feedback)
        return {"action": vector.action, "score": vector.score, "rewarded": rewarded}

    def run_once_from_wav(self, wav_path: str) -> dict[str, Any]:
        text = self.stt.transcribe(wav_path=wav_path)
        if not text:
            self.tts.speak("Команду не розпізнано")
            return {"action": "NONE", "score": 0.0, "rewarded": False}
        return self.handle_command(text)

