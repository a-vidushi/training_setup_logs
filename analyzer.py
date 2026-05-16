"""Trajectory validation and complexity tagging for agent sessions."""
from __future__ import annotations

from types_def import LogSession, TrainingMetadata

# ── Tool name allowlist ───────────────────────────────────────────────────────
# Set to None to skip allowlist checking.
ALLOWED_TOOL_NAMES: set[str] | None = {
    "weather_forecast",
    "fetch_agristack_data",
    "crop_recommendation",
    "subsidy_lookup",
    "market_price",
    "pest_advisory",
    "soil_report",
    "irrigation_schedule",
    #can keep adding more tool names here
}

# Words that signal an ambiguous or under-specified question
_AMBIGUITY_TOKENS: set[str] = {
    # English
    "what", "how", "when", "why", "which", "where", "any", "some",
    "maybe", "possibly", "sometimes", "generally",
    # Hindi / transliterated
    "kya", "kab", "kyun", "kaise", "kaun", "kuch", "koi",
    # Marathi
    "kay", "keva", "kasa",
}


def validate_trajectory(session: LogSession) -> bool:
    """Return True if the agent trajectory is structurally valid.

    Rules
    -----
    - tool-call parts must have a non-empty ``tool_name``.
    - If ``ALLOWED_TOOL_NAMES`` is set, ``tool_name`` must be in the allowlist.
    - tool-return parts must have a matching tool-call (by ``tool_call_id``)
      appearing earlier in the sequence.
    - tool-return with ``content = None`` is rejected; ``content = ""`` is
      accepted (explicit empty return is a valid signal).
    """
    if not session.agent_turns:
        return False  # Reject completely empty or malformed sessions (no meaningful trajectory)

    tool_calls_seen: set[str] = set()

    for turn in session.agent_turns:
        for part in turn.parts:
            if part.part_kind == "tool-call":
                if not part.tool_name:
                    return False
                if (
                    ALLOWED_TOOL_NAMES is not None
                    and part.tool_name not in ALLOWED_TOOL_NAMES
                ):
                    return False
                if part.tool_call_id:
                    tool_calls_seen.add(part.tool_call_id)

            elif part.part_kind == "tool-return":
                # content=None means the field was never set — reject
                # content=""  means the tool explicitly returned nothing — accept
                if part.content is None:
                    return False
                if part.tool_call_id and part.tool_call_id not in tool_calls_seen:
                    return False

    return True


def _ambiguity_score(text: str) -> float:
    """Heuristic ambiguity score in [0, 1].

    Counts the fraction of tokens that are in :data:`_AMBIGUITY_TOKENS`.
    """
    if not text:
        return 0.0
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t.strip("?.,!") in _AMBIGUITY_TOKENS)
    return round(min(hits / len(tokens), 1.0), 3)


def tag_complexity(session: LogSession, max_tokens: int = 8192) -> TrainingMetadata:
    """Compute and return :class:`TrainingMetadata` for *session*.

    Complexity tiers
    ----------------
    ``simple``
        0 tool calls — instruction-following warmup data.
    ``moderate``
        1–3 tool calls with no error recovery — main SFT bulk.
    ``complex``
        4+ tool calls **or** any recovery step — late-stage / DPO up-weight.
    """
    num_tools = 0
    unique_tools: set[str] = set()
    has_recovery = False
    total_tokens = 0

    for turn in session.agent_turns:
        if turn.usage:
            total_tokens += (
                turn.usage.get("input_tokens", 0) + turn.usage.get("output_tokens", 0)
            )

        for p in turn.parts:
            if p.part_kind == "tool-call":
                num_tools += 1
                if p.tool_name:
                    unique_tools.add(p.tool_name)
            if p.part_kind == "tool-return" and p.content:
                content_lower = str(p.content).lower()
                if any(
                    err in content_lower
                    for err in ["error", "timeout", "failed", "no results"]
                ):
                    has_recovery = True

    # Revised tier thresholds
    if num_tools == 0:
        complexity = "simple"
    elif num_tools <= 3 and not has_recovery:
        complexity = "moderate"
    else:
        complexity = "complex"

    student_eligible = not (num_tools > 4 or total_tokens > max_tokens)

    # multi_turn_depth: number of distinct user turns (always ≥1)
    # In the current schema every session is a single user question; this
    # counter is a placeholder for future multi-turn schema support.
    multi_turn_depth = 1

    return TrainingMetadata(
        tool_count=num_tools,
        unique_tools=list(unique_tools),
        has_recovery=has_recovery,
        complexity_tier=complexity,
        is_agentic=num_tools > 0,
        total_tokens=total_tokens,
        student_eligible=student_eligible,
        multi_turn_depth=multi_turn_depth,
        ambiguity_score=_ambiguity_score(session.user_question),
    )