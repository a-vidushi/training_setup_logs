"""PII redaction using Microsoft Presidio + custom Indian-ID recognizers.

Supported entity types:
    PHONE_NUMBER, EMAIL_ADDRESS, URL,
    IN_AADHAAR  – 12-digit Aadhaar (grouped or plain)
    IN_PAN      – 10-character PAN card  (AAAAA9999A)
    IN_VOTER_ID – 10-character Voter ID  (AAA9999999)

For non-English sessions the same patterns are applied via regex directly
(Presidio's NLP pipeline is English-only).
"""
import re
import collections

import spacy

try:
    spacy.util.get_package_path("en_core_web_lg")
except Exception:
    raise ImportError(
        "Missing required spacy model. "
        "Run: init_env.sh  or  python -m spacy download en_core_web_lg"
    )

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern  # type: ignore

analyzer = AnalyzerEngine()

# ── Indian-specific recognizers ──────────────────────────────────────────────

# Aadhaar: 12 digits optionally grouped as XXXX-XXXX-XXXX or XXXX XXXX XXXX
_AADHAAR_PATTERN = Pattern(
    name="aadhaar",
    regex=r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    score=0.85,
)
analyzer.registry.add_recognizer(
    PatternRecognizer(supported_entity="IN_AADHAAR", patterns=[_AADHAAR_PATTERN])
)

# PAN card: 5 uppercase letters, 4 digits, 1 uppercase letter
_PAN_PATTERN = Pattern(
    name="pan",
    regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    score=0.9,
)
analyzer.registry.add_recognizer(
    PatternRecognizer(supported_entity="IN_PAN", patterns=[_PAN_PATTERN])
)

# Voter ID (EPIC): 3 uppercase letters followed by 7 digits
_VOTER_PATTERN = Pattern(
    name="voter_id",
    regex=r"\b[A-Z]{3}[0-9]{7}\b",
    score=0.8,
)
analyzer.registry.add_recognizer(
    PatternRecognizer(supported_entity="IN_VOTER_ID", patterns=[_VOTER_PATTERN])
)

# Phone: +91, 0, or plain 10 digits starting with 6-9
_PHONE_PATTERN = Pattern(
    name="indian_phone",
    regex=r"\b(?:\+?91[- ]?)?(?:0[- ]?)?[6789]\d{2}[- ]?\d{3}[- ]?\d{4}\b",
    score=0.9,
)
analyzer.registry.add_recognizer(
    PatternRecognizer(supported_entity="IN_PHONE_NUMBER", patterns=[_PHONE_PATTERN])
)

# Entities to detect — PERSON excluded to avoid false positives on crop/place names
ENTITIES = [
    "IN_PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "URL",
    "IN_AADHAAR",
    "IN_PAN",
    "IN_VOTER_ID",
]

# ── Regex patterns for non-English (Presidio NLP is English-only) ────────────
_NON_EN_PATTERNS: list[tuple[str, str]] = [
    ("IN_PHONE_NUMBER",  r"\b(?:\+?91[- ]?)?(?:0[- ]?)?[6789]\d{2}[- ]?\d{3}[- ]?\d{4}\b"),
    ("IN_AADHAAR",   r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    ("EMAIL_ADDRESS", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    ("IN_PAN",       r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    ("IN_VOTER_ID",  r"\b[A-Z]{3}[0-9]{7}\b"),
]


def redact_pii(text: str, language: str = "en", _counts: dict | None = None):
    """Redact PII in *text*.

    Parameters
    ----------
    text:
        Raw text to redact.
    language:
        BCP-47 primary subtag (e.g. ``"en"``, ``"hi"``).  When ``"en"``,
        Presidio's full NLP pipeline is used.  For all other languages a
        regex-only fallback is applied.
    _counts:
        Shared counter dict for consistent placeholder numbering within a
        session.  Pass the *same* dict for every call in one session.

    Returns
    -------
    (redacted_text, audit_entries)
        ``audit_entries`` is a list of dicts with keys
        ``original_span``, ``placeholder``, ``entity_type``.
    """
    if not text:
        return text, []

    if _counts is None:
        _counts = collections.defaultdict(int)

    audit_entries: list[dict] = []

    if language != "en":
        # Regex-only path for non-English text
        matches: list[tuple[str, int, int]] = []
        for entity_type, pattern in _NON_EN_PATTERNS:
            for m in re.finditer(pattern, text):
                matches.append((entity_type, m.start(), m.end()))
                
        # Filter out overlapping entities (keep the longest span)
        # Sort by start index ascending, then length descending
        sorted_matches = sorted(matches, key=lambda x: (x[1], -(x[2] - x[1])))
        filtered_matches = []
        for m in sorted_matches:
            if not filtered_matches or m[1] >= filtered_matches[-1][2]:
                filtered_matches.append(m)

        # Sort right-to-left so replacements don't shift offsets
        filtered_matches.sort(key=lambda x: x[1], reverse=True)

        redacted_text = text
        for r_type, start, end in filtered_matches:
            entity_value = text[start:end]
            _counts[r_type] += 1
            placeholder = f"<{r_type}_{_counts[r_type]}>"
            audit_entries.append(
                {"original_span": entity_value, "placeholder": placeholder, "entity_type": r_type}
            )
            redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]
        return redacted_text, audit_entries

    # Full Presidio NLP path for English
    results = analyzer.analyze(text=text, entities=ENTITIES, language="en")
    
    # Filter out overlapping entities (keep the longest span)
    # Sort by start index ascending, then length descending
    sorted_results = sorted(results, key=lambda x: (x.start, -(x.end - x.start)))
    filtered_results = []
    for r in sorted_results:
        if not filtered_results or r.start >= filtered_results[-1].end:
            filtered_results.append(r)
            
    # Sort right-to-left for safe replacement
    filtered_results = sorted(filtered_results, key=lambda x: x.start, reverse=True)
    redacted_text = text

    for r in filtered_results:
        entity_value = text[r.start : r.end]
        _counts[r.entity_type] += 1
        placeholder = f"<{r.entity_type}_{_counts[r.entity_type]}>"
        audit_entries.append(
            {"original_span": entity_value, "placeholder": placeholder, "entity_type": r.entity_type}
        )
        redacted_text = redacted_text[: r.start] + placeholder + redacted_text[r.end :]

    return redacted_text, audit_entries