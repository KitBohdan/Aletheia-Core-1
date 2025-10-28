from vct.ethics.guard import EthicsGuard, EthicsConfig

def test_guard_cooldown_and_threshold():
    g = EthicsGuard(EthicsConfig(min_inter_reward_s=0.1))
    assert not g.can_reward(now_ts=0.0, action="SIT", score=0.7, cooldown_s=0.2)
    assert g.can_reward(now_ts=1.0, action="SIT", score=0.7, cooldown_s=0.2)
    g.note_reward(1.0)
    assert not g.can_reward(now_ts=1.05, action="SIT", score=1.0, cooldown_s=0.2)
