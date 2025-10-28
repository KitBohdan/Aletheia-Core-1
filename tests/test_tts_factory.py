
from vct.engines.tts import create_tts_engine, TTSConfig

def test_tts_factory_auto_has_speak():
    engine = create_tts_engine({"provider": "auto"})
    assert hasattr(engine, "speak")
    engine.speak("ok")
