"""Reward actuator implementations."""

from __future__ import annotations

import time
from typing import Any, Final


class RewardActuatorBase:
    """Base class for reward actuators."""

    def trigger(self, seconds: float = 0.5) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class SimulatedActuator(RewardActuatorBase):
    """Simulated dispenser for testing environments."""

    def trigger(self, seconds: float = 0.5) -> None:
        print(f"[REWARD] Simulated dispenser {seconds:.2f}s")
        time.sleep(min(seconds, 0.05))


class GPIOActuator(RewardActuatorBase):
    """Physical GPIO dispenser, falling back to the simulator when unavailable."""

    def __init__(self, pin: int) -> None:
        self.device: Any | None
        self.available: bool
        try:  # pragma: no cover - optional dependency on gpiozero
            from gpiozero import OutputDevice  # type: ignore

            self.device = OutputDevice(pin)
            self.available = True
        except Exception:  # pragma: no cover - hardware not available
            self.device = None
            self.available = False

    def trigger(self, seconds: float = 0.5) -> None:
        if not self.available or self.device is None:
            SimulatedActuator().trigger(seconds)
            return
        assert self.device is not None
        self.device.on()
        time.sleep(seconds)
        self.device.off()


DEFAULT_GPIO_PIN: Final[int] = 18
