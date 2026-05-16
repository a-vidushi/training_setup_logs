"""Synthetic data generation for the OpenAgriNet training pipeline.

Three generators are provided:

1. ``generate_hard_case``    — inserts deliberate ambiguity into the question
   to create harder training examples.
2. ``generate_failure_correction`` — creates a session where an intermediate
   tool-return is an error and the bot_response demonstrates correct recovery.

All synthetic sessions have ``source="synthetic"`` and a new ``session_id``.
"""
from __future__ import annotations

import copy
import hashlib
import random
import uuid

from types_def import AgentTurn, LogSession, Part

# ── Hard case generation ─────────────────────────────────────────────────────

_AMBIGUITY_PREFIXES = {
    "en": [
        "Generally speaking, ",
        "In some cases, ",
        "I'm not sure but, ",
        "Maybe you can tell me — ",
    ],
    "hi": [
        "सामान्य तौर पर, ",
        "क्या आप बता सकते हैं कि ",
        "मुझे पक्का नहीं पता लेकिन, ",
    ]
}

_VAGUE_SUFFIXES = {
    "en": [
        " (not sure which one)",
        " — any thoughts?",
        " if possible",
        " for my area",
        " approximately",
    ],
    "hi": [
        " (मुझे पक्का नहीं पता)",
        " — आपके क्या विचार हैं?",
        " यदि संभव हो",
        " मेरे क्षेत्र के लिए",
        " लगभग",
    ]
}


def generate_hard_case(session: LogSession) -> LogSession:
    """Return a copy of *session* whose user question is made more ambiguous."""
    rng = random.Random(
        int(hashlib.md5(session.user_question.encode()).hexdigest(), 16) & 0xFFFF_FFFF
    )
    lang = session.language if session.language in _AMBIGUITY_PREFIXES else "en"
    prefix = rng.choice(_AMBIGUITY_PREFIXES[lang])
    suffix = rng.choice(_VAGUE_SUFFIXES[lang])

    new_session = copy.deepcopy(session)
    new_session.session_id = str(uuid.uuid4())
    new_session.user_question = prefix + session.user_question.rstrip("?") + suffix
    new_session.source = "synthetic"
    return new_session


# ── Failure-correction generation ────────────────────────────────────────────

_RECOVERY_RESPONSES = [
    "I'm sorry, but I was unable to retrieve the live data at this time due to a temporary service issue. Please try again later.",
    "The required data service is currently unresponsive. I cannot provide the specific details you asked for right now.",
    "I encountered a connectivity issue while fetching your records. Please verify your details or try again in a few minutes.",
]

_ERROR_RETURN_CONTENTS = [
    "error: timeout fetching farmer record",
    "error: service unavailable (503)",
    "error: no results found for the given query",
]


def generate_failure_correction(session: LogSession) -> LogSession | None:
    """Return a copy of *session* where one tool-return is replaced with an
    error, and the bot_response is rewritten as a graceful recovery.

    Returns ``None`` if the session has no tool-return parts to corrupt.
    """
    new_session = copy.deepcopy(session)
    new_session.session_id = str(uuid.uuid4())
    new_session.source = "synthetic"

    rng = random.Random(
        int(hashlib.md5(session.user_question.encode()).hexdigest(), 16) & 0xFFFF_FFFF
    )

    # Find the first tool-return part and inject an error
    corrupted = False
    for turn in new_session.agent_turns:
        for part in turn.parts:
            if part.part_kind == "tool-return" and not corrupted:
                part.content = rng.choice(_ERROR_RETURN_CONTENTS)
                corrupted = True
                break
        if corrupted:
            break

    if not corrupted:
        return None

    new_session.bot_response = rng.choice(_RECOVERY_RESPONSES)
    return new_session
