from __future__ import annotations

import pytest

from vct.robodog.dog_bot_brain import RoboDogBrain
from vct.simulation.dog_env import DogEnv


class DummyBrain:
    def __init__(self) -> None:
        self.calls: list[dict[str, float | bool | str | None]] = []

    def handle_command(
        self,
        text: str,
        confidence: float = 0.85,
        reward_bias: float = 0.5,
        mood: float | None = None,
        energy_level: float | None = None,
    ) -> dict[str, float | bool | str | None]:
        self.calls.append(
            {
                "text": text,
                "confidence": confidence,
                "reward_bias": reward_bias,
                "mood": mood,
                "energy_level": energy_level,
            }
        )
        return {"action": "SIT", "score": 0.8, "rewarded": False}


def test_dog_env_passes_state_to_brain() -> None:
    env = DogEnv(seed=123)
    env.s.mood = 0.25
    env.s.fatigue = 0.6
    brain = DummyBrain()

    result = env.interact(brain, "сидіти", confidence=0.9, reward_bias=0.4)

    assert result["brain"]["action"] == "SIT"
    assert result["env"]["action"] == "SIT"
    assert brain.calls, "brain should have been invoked"
    call = brain.calls[0]
    assert call["mood"] == pytest.approx(0.25)
    assert call["energy_level"] == pytest.approx(0.4)


def test_dog_env_closed_loop_with_robo_brain() -> None:
    env = DogEnv(seed=1)
    brain = RoboDogBrain(cfg_path="vct/config.yaml", simulate=True)

    fatigue_values: list[float] = []
    for _ in range(5):
        step_result = env.interact(brain, "сидіти", confidence=0.9, reward_bias=0.6)
        brain_out = step_result["brain"]
        env_out = step_result["env"]
        assert isinstance(brain_out["score"], float)
        assert brain_out["action"] in {"SIT", "NONE"}
        assert 0.0 <= env_out["fatigue"] <= 1.0
        assert -1.0 <= env_out["mood"] <= 1.0
        fatigue_values.append(env_out["fatigue"])

    assert all(b >= a for a, b in zip(fatigue_values, fatigue_values[1:]))
