from vct.hardware.gpio_reward import GPIOActuator, SimulatedActuator


def test_simulated_actuator_runs() -> None:
    SimulatedActuator().trigger(0.01)


def test_gpio_actuator_fallback() -> None:
    actuator = GPIOActuator(pin=18)
    actuator.trigger(0.01)
