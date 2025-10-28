"""Speech-to-text engine implementations."""

from __future__ import annotations

import threading
import warnings
from pathlib import Path
from typing import Any, Callable


class STTEngineBase:
    """Abstract base class for speech-to-text engines."""

    def transcribe(self, wav_path: Path | None = None, use_mic: bool = False) -> str:
        """Convert speech to text from either a WAV file or microphone input."""

        raise NotImplementedError


class WhisperSTT(STTEngineBase):
    """Wrapper around the OpenAI Whisper model for speech recognition."""

    def __init__(
        self,
        model_name: str = "base",
        device: str | None = None,
        model_loader: Callable[[str, str | None], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model_loader: Callable[[str, str | None], Any] = (
            model_loader or self._default_model_loader
        )
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _default_model_loader(self, model_name: str, device: str | None) -> Any:
        import whisper

        load_kwargs: dict[str, Any] = {}
        if device is not None:
            load_kwargs["device"] = device
        return whisper.load_model(model_name, **load_kwargs)

    def _ensure_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._model_loader(self.model_name, self.device)
        return self._model

    def transcribe(self, wav_path: Path | None = None, use_mic: bool = False) -> str:
        if use_mic:
            raise NotImplementedError("Microphone transcription is not implemented for WhisperSTT")
        if not wav_path:
            return ""
        model = self._ensure_model()
        result = model.transcribe(str(wav_path))
        text: str
        if isinstance(result, dict):
            text = str(result.get("text", ""))
        else:
            text = str(result)
        return text.strip()


class RuleBasedSTT(WhisperSTT):
    """Backward-compatible alias for the legacy rule-based STT implementation."""

    def __init__(
        self,
        model_name: str = "base",
        device: str | None = None,
        model_loader: Callable[[str, str | None], Any] | None = None,
    ) -> None:
        warnings.warn(
            "RuleBasedSTT has been replaced by WhisperSTT. "
            "Please update your code to instantiate WhisperSTT directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(model_name=model_name, device=device, model_loader=model_loader)
