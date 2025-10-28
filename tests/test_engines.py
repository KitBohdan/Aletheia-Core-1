import sys

import pytest

from vct.engines.stt import WhisperSTT, RuleBasedSTT
from vct.engines.tts import PrintTTS, create_tts_engine


class _DummyWhisperModel:
    def __init__(self, transcript: str = "") -> None:
        self.transcript = transcript
        self.seen_paths = []

    def transcribe(self, wav_path: str):
        self.seen_paths.append(wav_path)
        return {"text": self.transcript}


def test_whisper_stt_uses_model_loader(tmp_path):
    dummy_model = _DummyWhisperModel("сидіти")
    stt = WhisperSTT(model_loader=lambda *_: dummy_model)
    fake_wav = tmp_path / "command.wav"
    fake_wav.write_bytes(b"not really audio")

    assert stt.transcribe(wav_path=fake_wav) == "сидіти"
    assert dummy_model.seen_paths == [str(fake_wav)]


def test_rulebased_stt_alias_emits_warning(tmp_path):
    dummy_model = _DummyWhisperModel("лежати")
    with pytest.deprecated_call():
        stt = RuleBasedSTT(model_loader=lambda *_: dummy_model)
    fake_wav = tmp_path / "command.wav"
    fake_wav.write_bytes(b"still not audio")
    assert stt.transcribe(wav_path=fake_wav) == "лежати"

def test_tts_print_ok():
    tts = PrintTTS()
    tts.speak("Тест")


def test_create_tts_engine_unknown_provider():
    engine = create_tts_engine({"provider": "does-not-exist"})
    assert isinstance(engine, PrintTTS)


def test_create_tts_engine_gtts_fallback(monkeypatch):
    class _BrokenModule:
        pass

    monkeypatch.setitem(sys.modules, "gtts", _BrokenModule())
    engine = create_tts_engine({"provider": "gtts"})
    assert isinstance(engine, PrintTTS)
