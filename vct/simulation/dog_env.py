"""Simple reinforcement learning environment for the RoboDog brain."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class EnvState:
    fatigue: float = 0.0
    mood: float = 0.0
    reward_history: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "fatigue": float(self.fatigue),
            "mood": float(self.mood),
            "reward_hist": float(self.reward_history),
        }


class DogEnv:
    """Lightweight environment modelling the RoboDog internal state."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self.s = EnvState()

    def observe(self) -> EnvState:
        """Return the current environment state."""

        return self.s

    def reset(self) -> EnvState:
        """Reset the environment to the neutral state."""

        self.s = EnvState()
        return self.s

    def step(self, action: str, score: float) -> dict[str, Any]:
        """Update the environment using the outcome produced by the brain."""

        success_p = 0.5 + 0.4 * score - 0.2 * self.s.fatigue
        success = self._rng.random() < _clamp(success_p, 0.05, 0.95)

        fatigue_delta = 0.1 if success else 0.05
        self.s.fatigue = _clamp(self.s.fatigue + fatigue_delta, 0.0, 1.0)

        mood_delta = 0.1 if success else -0.05
        self.s.mood = _clamp(self.s.mood + mood_delta, -1.0, 1.0)

        history_target = 1.0 if success else 0.0
        self.s.reward_history = self.s.reward_history * 0.8 + history_target * 0.2

        return {"success": success, **self.s.to_dict(), "action": action, "score": score}

    def interact(
        self,
        brain: Any,
        command: str,
        *,
        confidence: float = 0.85,
        reward_bias: float = 0.5,
    ) -> dict[str, Any]:
        """Run a single perception-action cycle with the provided brain."""

        brain_output: Mapping[str, Any] = brain.handle_command(
            command,
            confidence=confidence,
            reward_bias=reward_bias,
            mood=self.s.mood,
            energy_level=_clamp(1.0 - self.s.fatigue, 0.0, 1.0),
        )
        env_feedback = self.step(str(brain_output["action"]), float(brain_output["score"]))
        return {"brain": dict(brain_output), "env": env_feedback}
