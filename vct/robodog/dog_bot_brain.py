import time, yaml
from typing import Optional, Dict
from ..engines.stt import WhisperSTT
from ..engines.tts import create_tts_engine, PrintTTS
from ..hardware.gpio_reward import GPIOActuator, SimulatedActuator
from ..behavior.policy import BehaviorPolicy, BehaviorInputs
from ..ethics.guard import EthicsGuard
from ..utils.logging import get_logger

log = get_logger("RoboDogBrain")

class RoboDogBrain:
    def __init__(self, cfg_path: str, gpio_pin: Optional[int] = None, simulate: bool = False):
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.stt = WhisperSTT()
        if simulate:
            self.tts = PrintTTS()
        else:
            self.tts = create_tts_engine(self.cfg.get("tts"))
        self.policy = BehaviorPolicy(self.cfg.get("weights", {}))
        self.reward_map: Dict[str, bool] = self.cfg.get("reward_triggers", {})
        self.cooldown_s = float(self.cfg.get("reward_cooldown_s", 3))
        self.simulate = simulate
        self.actuator = SimulatedActuator() if (simulate or gpio_pin is None) else GPIOActuator(gpio_pin)
        self.guard = EthicsGuard()
        defaults = self.cfg.get("behavior_defaults", {})
        self.behavior_defaults = {
            "energy_level": float(defaults.get("energy_level", 0.6)),
            "proximity": float(defaults.get("proximity", 0.5)),
            "threat_level": float(defaults.get("threat_level", 0.1)),
            "social_context": float(defaults.get("social_context", 0.5)),
        }
        context_defaults = defaults.get("context", {})
        if isinstance(context_defaults, dict):
            self.behavior_context = {k: float(v) for k, v in context_defaults.items()}
        else:
            self.behavior_context = {}

    def _action_from_text(self, text: str) -> str:
        m = self.cfg.get("commands_map", {})
        n = text.strip().lower().replace(" ", "")
        for k, v in m.items():
            if k.replace(" ", "") in n:
                return v
        return "NONE"

    def _maybe_reward(self, action: str, score: float) -> bool:
        if not self.reward_map.get(action, False):
            return False
        now = time.time()
        if not self.guard.can_reward(now, action, score, self.cooldown_s):
            return False
        self.actuator.trigger(0.4)
        self.guard.note_reward(now)
        return True

    def handle_command(self, text: str, confidence: float = 0.85, reward_bias: float = 0.5, mood: float = 0.0) -> Dict:
        action = self._action_from_text(text)
        context = dict(self.behavior_context)
        context["action_known"] = 1.0 if action != "NONE" else 0.0
        context["reward_available"] = 1.0 if self.reward_map.get(action, False) else 0.0
        inputs = BehaviorInputs(
            stimulus=1.0 if action != "NONE" else 0.0,
            confidence=confidence,
            reward_bias=reward_bias,
            mood=mood,
            energy_level=self.behavior_defaults["energy_level"],
            proximity=self.behavior_defaults["proximity"],
            threat_level=self.behavior_defaults["threat_level"],
            social_context=self.behavior_defaults["social_context"],
            context=context,
        )
        vec = self.policy.decide(action, inputs)
        rewarded = self._maybe_reward(vec.action, vec.score)
        feedback = f"Дія: {vec.action} score={vec.score:.2f}" + (" — ✅ винагорода" if rewarded else "")
        self.tts.speak(feedback); log.info(feedback)
        return {"action": vec.action, "score": vec.score, "rewarded": rewarded}

    def run_once_from_wav(self, wav_path: str) -> Dict:
        text = self.stt.transcribe(wav_path=wav_path)
        if not text:
            self.tts.speak("Команду не розпізнано")
            return {"action": "NONE", "score": 0.0, "rewarded": False}
        return self.handle_command(text)