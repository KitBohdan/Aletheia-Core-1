
from vct.simulation.dog_env import DogEnv

class DummyBrain:
    def handle_command(self, *args, **kwargs):
        return {"action":"SIT", "score":0.8}

def test_interact_flow():
    env = DogEnv()
    env.reset()
    out = env.interact(DummyBrain(), "сидіти", confidence=0.9, reward_bias=0.4)
    assert isinstance(out, dict)
    assert "brain" in out
