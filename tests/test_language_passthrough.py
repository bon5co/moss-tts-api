"""Tests for the `language` passthrough on the TTS path.

These mock the processor entirely — no model weights are downloaded and no
generation runs. What is under test is exactly which keyword arguments reach
`processor.build_user_message`.
"""

from __future__ import annotations

import logging

import pytest

from app import engine as engine_mod
from app.config import settings
from app.engine import _build_user_message
from app.routes import SpeechRequest, VoiceDesignRequest


class FakeProcessor:
    """Records the kwargs of every build_user_message call."""

    def __init__(self):
        self.calls: list[dict] = []

    def build_user_message(self, *, text, reference=None, language=None):
        call = {"text": text, "reference": reference}
        if language is not None:
            call["language"] = language
        self.calls.append(call)
        return call


class LegacyProcessor:
    """An older revision whose build_user_message has no `language` param."""

    def __init__(self):
        self.calls: list[dict] = []

    def build_user_message(self, *, text, reference=None):
        call = {"text": text, "reference": reference}
        self.calls.append(call)
        return call


class RejectingProcessor:
    """Signature claims **kwargs but the body rejects `language`."""

    def __init__(self):
        self.calls: list[dict] = []

    def build_user_message(self, **kwargs):
        if "language" in kwargs:
            raise TypeError("unexpected keyword argument 'language'")
        self.calls.append(kwargs)
        return kwargs


@pytest.fixture(autouse=True)
def _no_default_language(monkeypatch):
    """Isolate tests from any DEFAULT_LANGUAGE in the ambient environment."""
    monkeypatch.setattr(settings, "default_language", None)


# --- (a) language omitted -> called WITHOUT the language kwarg ---------------


def test_language_omitted_calls_without_language_kwarg():
    proc = FakeProcessor()
    _build_user_message(proc, text="hello", reference=None, language=None)
    assert proc.calls == [{"text": "hello", "reference": None}]
    assert "language" not in proc.calls[0]


def test_language_omitted_with_reference_matches_legacy_call():
    proc = FakeProcessor()
    _build_user_message(proc, text="hi", reference=["/tmp/ref.wav"], language=None)
    assert proc.calls == [{"text": "hi", "reference": ["/tmp/ref.wav"]}]


# --- (b) language="Japanese" -> called WITH it -------------------------------


def test_language_set_passes_language_kwarg():
    proc = FakeProcessor()
    _build_user_message(proc, text="こんにちは", reference=None, language="Japanese")
    assert proc.calls == [
        {"text": "こんにちは", "reference": None, "language": "Japanese"}
    ]


def test_arbitrary_language_name_passes_through_unvalidated():
    proc = FakeProcessor()
    _build_user_message(proc, text="bonjour", reference=None, language="French")
    assert proc.calls[0]["language"] == "French"


# --- robustness: processors that cannot take the kwarg -----------------------


def test_legacy_processor_falls_back_and_warns(caplog):
    proc = LegacyProcessor()
    with caplog.at_level(logging.WARNING, logger="moss-tts.engine"):
        _build_user_message(proc, text="hi", reference=None, language="Japanese")
    assert proc.calls == [{"text": "hi", "reference": None}]
    assert "does not accept `language`" in caplog.text


def test_typeerror_at_call_time_retries_without_language(caplog):
    proc = RejectingProcessor()
    with caplog.at_level(logging.WARNING, logger="moss-tts.engine"):
        _build_user_message(proc, text="hi", reference=None, language="Japanese")
    assert proc.calls == [{"text": "hi", "reference": None}]
    assert "rejected `language`" in caplog.text


# --- settings default --------------------------------------------------------


def test_default_language_is_none_by_default():
    """Unconfigured servers must behave exactly as before this change."""
    from app.config import Settings

    assert Settings(_env_file=None).default_language is None


def test_synthesize_applies_configured_default(monkeypatch):
    """A request without `language` picks up settings.default_language."""
    monkeypatch.setattr(settings, "default_language", "Japanese")
    seen = {}

    def fake_build(processor, *, text, reference, language):
        seen["language"] = language
        return None

    monkeypatch.setattr(engine_mod, "_build_user_message", fake_build)

    eng = engine_mod.Engine.__new__(engine_mod.Engine)
    import threading

    eng._lock = threading.Lock()
    eng._processor = FakeProcessor()
    eng.device = "cpu"
    monkeypatch.setattr(
        engine_mod.Engine, "ensure_loaded", lambda self, *a, **k: None
    )
    with pytest.raises(Exception):
        # Generation itself will fail on the fake processor; we only care
        # that the language resolution ran first.
        eng.synthesize("text")
    assert seen["language"] == "Japanese"


# --- request models -----------------------------------------------------------


def test_speech_request_language_is_optional():
    req = SpeechRequest(input="hello")
    assert req.language is None
    assert SpeechRequest(input="hello", language="Japanese").language == "Japanese"


def test_voice_design_request_has_no_language_field():
    """VoiceGenerator (1.7B) is Chinese/English only — no language field."""
    assert "language" not in VoiceDesignRequest.model_fields
    req = VoiceDesignRequest(input="hi", instruction="calm narrator", language="Japanese")
    assert not hasattr(req, "language")


def test_clone_endpoint_exposes_optional_language_form_field():
    import inspect

    from app.routes import clone_speech

    params = inspect.signature(clone_speech).parameters
    assert "language" in params
    assert params["language"].default.default is None


def test_design_voice_never_passes_language(monkeypatch):
    """The design_voice call site must not send a language kwarg.

    Even with a default language configured, the VoiceGenerator path must
    call build_user_message with text + instruction only.
    """
    monkeypatch.setattr(settings, "default_language", "Japanese")
    calls: list[dict] = []

    class StopHere(Exception):
        """Raised to abort before any real generation happens."""

    class DesignProcessor:
        def build_user_message(self, **kwargs):
            calls.append(kwargs)
            raise StopHere

    import threading

    eng = engine_mod.Engine.__new__(engine_mod.Engine)
    eng._lock = threading.Lock()
    eng._processor = DesignProcessor()
    eng.device = "cpu"
    monkeypatch.setattr(
        engine_mod.Engine, "ensure_loaded", lambda self, *a, **k: None
    )

    with pytest.raises(StopHere):
        eng.design_voice("hello", "calm narrator")

    assert calls == [{"text": "hello", "instruction": "calm narrator"}]
