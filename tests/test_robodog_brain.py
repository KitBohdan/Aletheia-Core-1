import pytest

from vct.robodog.dog_bot_brain import RoboDogBrain


pytestmark = pytest.mark.slow

def test_brain_handle_command_simulated():
    brain = RoboDogBrain(simulate=True)
    out = brain.handle_command("сидіти", 0.9, 0.5, 0.0)
    assert isinstance(out, dict)
    assert "action" in out and "score" in out
