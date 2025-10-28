from vct.behavior.policy import BehaviorInputs, BehaviorPolicy

def test_decide_scores_in_range():
    bp = BehaviorPolicy(weights={"stimulus": 0.4, "confidence": 0.3, "reward_bias": 0.2, "mood": 0.1})
    inp = BehaviorInputs(stimulus=0.8, confidence=0.9, reward_bias=0.5, mood=0.0)
    out = bp.decide("SIT", inp)
    assert 0.0 <= out.score <= 1.0
    assert out.action == "SIT"
