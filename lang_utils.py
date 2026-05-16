"""Language detection utility.

Uses langdetect (lightweight, offline) to infer a BCP-47 language code
from raw text.  The result is used by the PII redactor to choose the
correct analysis path and is cached onto LogSession.language when not
already set by the upstream system.
"""
from __future__ import annotations

SUPPORTED_LANGUAGES: set[str] = {"en", "hi", "mr", "pa", "te", "ta", "kn", "bn", "gu"}

_FALLBACK_LANG = "en"


def detect_language(text: str) -> str:
    """Return a language code for *text*.

    Falls back to ``"en"`` when detection fails or the detected language is
    outside :data:`SUPPORTED_LANGUAGES`.
    """
    if not text or not text.strip():
        return _FALLBACK_LANG
    try:
        from langdetect import detect, LangDetectException  # type: ignore
        lang = detect(text)
        # langdetect returns zh-cn etc.; normalise to primary subtag
        primary = lang.split("-")[0]
        return primary if primary in SUPPORTED_LANGUAGES else _FALLBACK_LANG
    except Exception:
        return _FALLBACK_LANG


def ensure_language(session) -> str:  # session: LogSession (avoid circular import)
    """Return session language, detecting it from user_question if absent."""
    if session.language:
        return session.language
    lang = detect_language(session.user_question)
    session.language = lang
    return lang
