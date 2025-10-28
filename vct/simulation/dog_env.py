"""Simple reinforcement learning environment for the RoboDog brain."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class EnvState:
    fatigue: float = 0.0
    mood: float = 0.0
    reward_history: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return {
            "fatigue": float(self.fatigue),
            "mood": float(self.mood),
            "reward_hist": float(self.reward_history),
        }


class DogEnv:
    """Lightweight environment modelling the RoboDog internal state.

    The simulator keeps track of fatigue, mood and reward history which evolve
    in response to the RoboDog brain outputs.  It can operate in a closed loop
    with :class:`vct.robodog.dog_bot_brain.RoboDogBrain` to quickly evaluate new
    behavioural policies without hardware in the loop.
    """

    def __init__(self, seed: int = 42):
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
        """Update the environment using the outcome produced by the brain.

        The probability of a successful action is influenced by the RoboDog's
        confidence (``score``) and accumulated fatigue.  Success improves the
        dog's mood while failures lower it and accumulate fatigue.
        """

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
        """Run a single perception-action cycle with the provided brain.

        The method feeds the current emotional state (mood) and a derived energy
        level (``1 - fatigue``) into :meth:`RoboDogBrain.handle_command` before
        updating the simulator state with the returned decision.
        """

        brain_output = brain.handle_command(
            command,
            confidence=confidence,
            reward_bias=reward_bias,
            mood=self.s.mood,
            energy_level=_clamp(1.0 - self.s.fatigue, 0.0, 1.0),
        )
        env_feedback = self.step(brain_output["action"], brain_output["score"])
        return {"brain": brain_output, "env": env_feedback}
