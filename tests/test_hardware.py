from vct.hardware.gpio_reward import SimulatedActuator

def test_simulated_actuator_trigger():
    SimulatedActuator().trigger(0.01)
