
from vct.ethics.guard import EthicsGuard

def test_guard_blocks_bark_by_default():
    g = EthicsGuard()
    assert not g.can_reward(10.0, "BARK", 0.99, 0.0)
