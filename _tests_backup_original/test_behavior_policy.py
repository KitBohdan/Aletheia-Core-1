from vct.behavior.policy import BehaviorInputs, BehaviorPolicy


def test_policy_learns_preferences():
    policy = BehaviorPolicy({}, hidden_size=6, learning_rate=0.1, random_seed=1, baseline_mix=0.1)
    training_data = [
        (
            BehaviorInputs(
                stimulus=1.0,
                confidence=0.9,
                reward_bias=0.8,
                mood=0.6,
                energy_level=0.9,
                proximity=0.3,
                threat_level=0.1,
                social_context=0.8,
                context={"owner_present": 1.0},
            ),
            0.9,
        ),
        (
            BehaviorInputs(
                stimulus=0.2,
                confidence=0.3,
                reward_bias=0.2,
                mood=-0.4,
                energy_level=0.2,
                proximity=0.8,
                threat_level=0.7,
                social_context=0.3,
                context={"owner_present": 0.0},
            ),
            0.1,
        ),
    ]
    policy.train(training_data, epochs=250)

    assert 0.0 <= policy.decide("FETCH", training_data[0][0]).score <= 1.0
    assert 0.0 <= policy.decide("STAY", training_data[1][0]).score <= 1.0

    high_score = policy.decide("FETCH", training_data[0][0]).score
    low_score = policy.decide("STAY", training_data[1][0]).score
    assert high_score > 0.7
    assert low_score < 0.4


def test_context_signal_impacts_score():
    policy = BehaviorPolicy({}, hidden_size=6, learning_rate=0.2, random_seed=2, baseline_mix=0.0)
    calm_environment = BehaviorInputs(
        stimulus=0.6,
        confidence=0.7,
        reward_bias=0.5,
        mood=0.2,
        energy_level=0.6,
        proximity=0.4,
        threat_level=0.2,
        social_context=0.6,
        context={"noise": 0.1, "owner_present": 1.0},
    )
    chaotic_environment = BehaviorInputs(
        stimulus=0.6,
        confidence=0.7,
        reward_bias=0.5,
        mood=0.2,
        energy_level=0.6,
        proximity=0.4,
        threat_level=0.2,
        social_context=0.6,
        context={"noise": 0.9, "owner_present": 0.0},
    )

    # Prefer calm environments with the handler present
    training_batch = []
    for _ in range(6):
        training_batch.append((calm_environment, 0.85))
    for _ in range(6):
        training_batch.append((chaotic_environment, 0.1))
    policy.train(training_batch, epochs=400)

    calm_score = policy.decide("FOLLOW", calm_environment).score
    chaotic_score = policy.decide("FOLLOW", chaotic_environment).score

    assert calm_score > chaotic_score
    assert calm_score - chaotic_score > 0.1
