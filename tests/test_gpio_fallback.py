
from vct.hardware.gpio_reward import GPIOActuator

def test_gpio_actuator_fallback_to_sim():
    gpio = GPIOActuator(pin=17)
    # Even if real GPIO isn't available here, trigger should not crash
    gpio.trigger(0.01)
