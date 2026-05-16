"""Rule-based persona scoring for the OpenAgriNet bot.

The bot persona is a helpful, honest, multilingual agricultural assistant.
This module checks the *bot_response* (and optionally the tool results)
against verifiable persona rules and returns a score in [0.0, 1.0].

Score interpretation
--------------------
≥ 0.8   High-quality response — include in both SFT and DPO chosen.
0.4–0.8 Acceptable — include in SFT only.
< 0.4   Poor quality — exclude from training; flag for human review.

The threshold for DPO chosen exclusion is 0.4 (see exporter.py).
"""
from __future__ import annotations

import re

from types_def import LogSession

# ── Penalty signals ──────────────────────────────────────────────────────────

# Over-refusal: bot refused a query it should have answered
_OVER_REFUSAL_PHRASES = [
    r"\bcannot help\b",
    r"\bunable to assist\b",
    r"\b(i|I) (don't|do not) know\b",
    r"\bsorry,? (i|I) (can't|cannot)\b",
]

# Hallucinated scheme names that do not exist in Indian agriculture
_HALLUCINATED_SCHEMES = [
    "pradhan mantri krishi samriddhi",  # not a real scheme
    "kisan welfare fund 2025",          # fictional
]

# Responses that lack any hedging on uncertain information
_HEDGE_SIGNALS = [
    "approximately", "around", "typically", "generally", "may vary",
    "consult", "check with", "subject to", "up to",
    # Hindi/Marathi equivalents (transliterated)
    "lagbhag", "samantan", "aaspaas",
]


def score_persona(session: LogSession) -> float:
    """Return a persona adherence score in [0.0, 1.0] for *session*.

    Scoring is additive-penalty: start at 1.0, deduct for each violation.
    """
    response = session.bot_response or ""
    response_lower = response.lower()
    score = 1.0

    # 1. Over-refusal penalty (-0.3 each, capped)
    refusal_hits = sum(
        1 for p in _OVER_REFUSAL_PHRASES if re.search(p, response_lower)
    )
    score -= min(refusal_hits * 0.3, 0.6)

    # 2. Hallucinated scheme penalty (-0.2 each)
    for scheme in _HALLUCINATED_SCHEMES:
        if scheme in response_lower:
            score -= 0.2

    # 3. Missing hedge on uncertain numeric claims (-0.1)
    # Heuristic: if response contains a number + "%" but no hedge word, penalise
    has_pct_claim = bool(re.search(r"\d+\s*%", response))
    has_hedge = any(h in response_lower for h in _HEDGE_SIGNALS)
    if has_pct_claim and not has_hedge:
        score -= 0.1

    # 4. Language mismatch penalty (-0.2)
    # If session is tagged as non-English but the bot responded entirely in
    # ASCII (likely English), penalise.
    # We skip this penalty if the user question was also transliterated (mostly ASCII).
    if session.language and session.language != "en":
        bot_non_ascii = sum(1 for c in response if ord(c) > 127) / max(len(response), 1)
        user_non_ascii = sum(1 for c in session.user_question if ord(c) > 127) / max(len(session.user_question), 1)
        if bot_non_ascii < 0.05 and user_non_ascii >= 0.05:  # bot is ASCII but user used native script
            score -= 0.2

    # 5. Empty response (-1.0)
    if not response.strip():
        score = 0.0

    return round(max(score, 0.0), 3)
