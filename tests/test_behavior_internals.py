
from vct.behavior.policy import BehaviorInputs, BehaviorPolicy

def test_feature_vector_and_decide():
    inp = BehaviorInputs(0.5, 0.7, 0.6, mood=0.1, energy_level=0.9, proximity=0.4, threat_level=0.0, social_context=0.3, context={"owner":1.0})
    vec = inp.to_feature_vector()
    assert len(vec) == len(BehaviorPolicy.feature_names)
    bp = BehaviorPolicy()
    out = bp.decide("COME", inp)
    assert 0.0 <= out.score <= 1.0
