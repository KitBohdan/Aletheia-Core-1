
from vct.simulation.dog_env import DogEnv

def test_dog_env_reset_and_step():
    env = DogEnv()
    s0 = env.reset()
    obs0 = env.observe()
    if hasattr(obs0, "to_dict"):
        d0 = obs0.to_dict()
    else:
        d0 = getattr(obs0, "__dict__", {})
    assert isinstance(d0, dict)
    s1 = env.step("SIT", 0.9)
    assert isinstance(s1, dict)
    assert "fatigue" in s1
