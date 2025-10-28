import pytest

from vct.engines.stt import WhisperSTT, RuleBasedSTT
from vct.engines.tts import PrintTTS


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
