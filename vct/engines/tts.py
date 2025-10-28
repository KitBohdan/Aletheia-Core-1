"""Utilities for text-to-speech output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


class TTSEngineBase:
    """Abstract interface for text to speech backends."""

    def speak(self, text: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def is_usable(self) -> bool:
        """Return whether the engine can actually synthesise audio."""

        return True


class Pyttsx3TTS(TTSEngineBase):
    """Local TTS backend relying on the ``pyttsx3`` library."""

    def __init__(self) -> None:
        try:  # pragma: no cover - depends on optional dependency
            import pyttsx3

            self.engine = pyttsx3.init()
        except Exception:  # pragma: no cover - graceful degradation
            self.engine = None

    def is_usable(self) -> bool:
        return self.engine is not None

    def speak(self, text: str) -> None:
        if self.engine is None:
            print(f"[TTS] {text}")
        else:  # pragma: no cover - depends on audio stack
            self.engine.say(text)
            self.engine.runAndWait()


class GTTSTTS(TTSEngineBase):
    """Cloud TTS backend using the `gTTS` API with multiple locales."""

    def __init__(
        self,
        language: str = "en",
        voice: str | None = None,
        slow: bool = False,
    ) -> None:
        self.language = language
        self.voice = voice or "com"
        self.slow = slow
        self._gtts_cls: type[Any] | None = None
        self._playsound: Any | None = None
        self._error: str | None = None
        try:  # pragma: no cover - depends on optional dependency
            from gtts import gTTS
            from playsound import playsound

            self._gtts_cls = gTTS
            self._playsound = playsound
        except Exception as exc:  # pragma: no cover - optional dependency
            self._error = str(exc)

    def is_usable(self) -> bool:
        return self._gtts_cls is not None and self._playsound is not None

    def speak(self, text: str) -> None:
        if not self.is_usable():
            if self._error:
                print(f"[TTS:gTTS disabled] {text} ({self._error})")
            else:
                print(f"[TTS] {text}")
            return

        assert self._gtts_cls is not None  # for mypy
        assert self._playsound is not None  # for mypy
        with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_path = Path(tmp_file.name)
            tts = self._gtts_cls(text=text, lang=self.language, tld=self.voice, slow=self.slow)
            tts.write_to_fp(tmp_file)

        try:  # pragma: no cover - depends on local audio stack
            self._playsound(str(tmp_path))
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


class PrintTTS(TTSEngineBase):
    """Fallback engine that prints spoken text to stdout."""

    def speak(self, text: str) -> None:
        print(f"[TTS] {text}")


@dataclass
class TTSConfig:
    """Configuration data for the TTS factory."""

    provider: str = "auto"
    language: str = "en"
    voice: str | None = None
    slow: bool = False

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any] | None) -> TTSConfig:
        if not cfg:
            return cls()
        return cls(
            provider=str(cfg.get("provider", "auto")).lower(),
            language=str(cfg.get("language", "en")),
            voice=cfg.get("voice"),
            slow=bool(cfg.get("slow", False)),
        )


def create_tts_engine(cfg: dict[str, Any] | None = None) -> TTSEngineBase:
    """Factory that returns the most appropriate TTS engine."""

    options = TTSConfig.from_mapping(cfg)
    provider = options.provider

    if provider in {"print", "console"}:
        return PrintTTS()
    if provider == "pyttsx3":
        pytt_engine = Pyttsx3TTS()
        return pytt_engine if pytt_engine.is_usable() else PrintTTS()
    if provider == "gtts":
        gtts_engine = GTTSTTS(language=options.language, voice=options.voice, slow=options.slow)
        return gtts_engine if gtts_engine.is_usable() else PrintTTS()

    if provider == "auto":
        gtts_engine = GTTSTTS(language=options.language, voice=options.voice, slow=options.slow)
        if gtts_engine.is_usable():
            return gtts_engine
        pytt_engine = Pyttsx3TTS()
        if pytt_engine.is_usable():
            return pytt_engine
        return PrintTTS()

    # Unknown provider - gracefully fall back to printing output.
    return PrintTTS()
