from vct.engines.tts import create_tts_engine, PrintTTS

def test_tts_factory_print():
    engine = create_tts_engine({"provider": "print"})
    assert isinstance(engine, PrintTTS)
    engine.speak("test line")
