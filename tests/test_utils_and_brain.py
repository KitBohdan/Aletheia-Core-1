
from vct.utils.logging import get_logger
from vct.robodog.dog_bot_brain import RoboDogBrain

def test_get_logger_idempotent():
    log1 = get_logger("vct.test")
    log2 = get_logger("vct.test")
    assert log1 is log2

def test_brain_init_with_overrides():
    brain = RoboDogBrain(simulate=True, config_overrides={"weights.stimulus": 0.3})
    out = brain.handle_command("голос", 0.8, 0.5, 0.0)
    assert "action" in out
