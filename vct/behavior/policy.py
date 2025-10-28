from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class BehaviorInputs:
    stimulus: float
    confidence: float
    reward_bias: float
    mood: float = 0.0  # -1..1
    energy_level: float = 0.5
    proximity: float = 0.5
    threat_level: float = 0.0
    social_context: float = 0.5
    context: Dict[str, float] = field(default_factory=dict)

    def context_signal(self) -> float:
        if not self.context:
            return 0.5
        values = [self._clamp(float(v)) for v in self.context.values()]
        return sum(values) / len(values)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def to_feature_vector(self) -> List[float]:
        mood_normalised = (self._clamp((self.mood + 1.0) / 2.0))
        base_features = [
            self._clamp(self.stimulus),
            self._clamp(self.confidence),
            self._clamp(self.reward_bias),
            mood_normalised,
            self._clamp(self.energy_level),
            self._clamp(self.proximity),
            self._clamp(self.threat_level),
            self._clamp(self.social_context),
        ]
        base_features.append(self.context_signal())
        return base_features


@dataclass
class BehaviorVector:
    score: float
    action: str


class BehaviorPolicy:
    feature_names: Sequence[str] = (
        "stimulus",
        "confidence",
        "reward_bias",
        "mood",
        "energy_level",
        "proximity",
        "threat_level",
        "social_context",
        "context_signal",
    )

    def __init__(
        self,
        weights: Dict[str, float] | None = None,
        *,
        hidden_size: int = 8,
        learning_rate: float = 0.05,
        random_seed: int = 42,
        baseline_mix: float = 0.2,
    ):
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.random_seed = random_seed
        self.input_size = len(self.feature_names)
        self.baseline_mix = max(0.0, min(1.0, baseline_mix))
        random.seed(random_seed)

        init_bound = 1.0 / math.sqrt(self.input_size)
        self.W1: List[List[float]] = [
            [random.uniform(-init_bound, init_bound) for _ in range(self.input_size)]
            for _ in range(self.hidden_size)
        ]
        self.b1: List[float] = [0.0 for _ in range(self.hidden_size)]
        self.W2: List[float] = [random.uniform(-init_bound, init_bound) for _ in range(self.hidden_size)]
        self.b2: float = 0.0

        self.default_weights: Dict[str, float] = {
            "stimulus": 0.4,
            "confidence": 0.3,
            "reward_bias": 0.2,
            "mood": 0.1,
            "energy_level": 0.15,
            "proximity": 0.1,
            "threat_level": 0.2,
            "social_context": 0.15,
            "context_signal": 0.15,
        }
        self.legacy_weights = {
            name: float((weights or {}).get(name, self.default_weights.get(name, 0.1)))
            for name in self.feature_names
        }

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _baseline_score(self, features: Sequence[float]) -> float:
        score = 0.0
        for idx, name in enumerate(self.feature_names):
            score += self.legacy_weights.get(name, 0.0) * features[idx]
        return max(0.0, min(1.0, score))

    def _forward(self, features: Sequence[float]) -> Tuple[List[float], float]:
        hidden: List[float] = []
        for i in range(self.hidden_size):
            activation = self.b1[i]
            weights = self.W1[i]
            for j, value in enumerate(features):
                activation += weights[j] * value
            hidden.append(math.tanh(activation))
        output_activation = self.b2
        for i, h in enumerate(hidden):
            output_activation += self.W2[i] * h
        return hidden, self._sigmoid(output_activation)

    def _backpropagate(self, features: Sequence[float], hidden: Sequence[float], output: float, target: float) -> None:
        error = output - target
        grad_W2: List[float] = [error * h for h in hidden]
        grad_b2 = error

        grad_hidden: List[float] = []
        for i in range(self.hidden_size):
            grad_hidden.append((1.0 - hidden[i] ** 2) * self.W2[i] * error)

        for i in range(self.hidden_size):
            for j in range(self.input_size):
                self.W1[i][j] -= self.learning_rate * grad_hidden[i] * features[j]
            self.b1[i] -= self.learning_rate * grad_hidden[i]

        for i in range(self.hidden_size):
            self.W2[i] -= self.learning_rate * grad_W2[i]
        self.b2 -= self.learning_rate * grad_b2

    def train(
        self,
        dataset: Iterable[Tuple[BehaviorInputs, float]],
        epochs: int = 50,
    ) -> None:
        data: List[Tuple[BehaviorInputs, float]] = list(dataset)
        if not data:
            return
        for _ in range(max(1, epochs)):
            random.shuffle(data)
            for inputs, target in data:
                target_clamped = max(0.0, min(1.0, float(target)))
                features = inputs.to_feature_vector()
                hidden, output = self._forward(features)
                self._backpropagate(features, hidden, output, target_clamped)

    def decide(self, action: str, inputs: BehaviorInputs) -> BehaviorVector:
        features = inputs.to_feature_vector()
        _, score_nn = self._forward(features)
        baseline = self._baseline_score(features)
        score = (1.0 - self.baseline_mix) * score_nn + self.baseline_mix * baseline
        score = max(0.0, min(1.0, score))
        return BehaviorVector(score=score, action=action)
