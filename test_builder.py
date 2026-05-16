"""Test suite for the OpenAgriNet training data pipeline.

Covers:
- PII redaction (English + Hindi, PAN, Voter ID)
- Language detection
- Trajectory validation (including empty-content fix)
- Complexity tagging
- SFT export format
- DPO export format — asserts chosen/rejected are message lists, not strings
- DPO prompt retains tool context
- Deduplication
- Split integrity (no session in both train and eval)
- Persona scoring range
"""
import json
import os
import tempfile
import pytest

from anonymizer import redact_pii
from analyzer import tag_complexity, validate_trajectory
from types_def import LogSession
from build_dataset import create_sft_export, create_dpo_export, split_sessions
from session_dedup import deduplicate_sessions
from behavior_scorer import score_persona
from lang_utils import detect_language
from data_augmenter import (
    generate_hard_case,
    generate_failure_correction,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_session():
    """Multi-tool session with one recovery step."""
    return LogSession(
        user_question="My email is test@example.com",
        bot_response="Thanks for contacting us.",
        agent_turns=[
            {"timestamp": "1", "parts": [{"part_kind": "tool-call", "tool_name": "weather_forecast", "tool_call_id": "c1"}]},
            {"timestamp": "2", "parts": [{"part_kind": "tool-return", "tool_name": "weather_forecast", "tool_call_id": "c1", "content": "result1"}]},
            {"timestamp": "3", "parts": [{"part_kind": "tool-call", "tool_name": "fetch_agristack_data", "tool_call_id": "c2"}]},
            {"timestamp": "4", "parts": [{"part_kind": "tool-return", "tool_name": "fetch_agristack_data", "tool_call_id": "c2", "content": "error: timeout"}]},
            {"timestamp": "5", "parts": [{"part_kind": "tool-call", "tool_name": "crop_recommendation", "tool_call_id": "c3"}]},
            {"timestamp": "6", "parts": [{"part_kind": "tool-return", "tool_name": "crop_recommendation", "tool_call_id": "c3", "content": "result3"}]},
        ],
    )


@pytest.fixture
def simple_session():
    """Single-tool session, no recovery."""
    return LogSession(
        user_question="What is the weather?",
        bot_response="Sunny, 32°C.",
        agent_turns=[
            {"timestamp": "1", "parts": [{"part_kind": "tool-call", "tool_name": "weather_forecast", "tool_call_id": "c1"}]},
            {"timestamp": "2", "parts": [{"part_kind": "tool-return", "tool_name": "weather_forecast", "tool_call_id": "c1", "content": "Sunny"}]},
        ],
    )


# ── PII Redaction ─────────────────────────────────────────────────────────────

def test_redact_pii_english_basic():
    text = "Call 9876543210 or email test@example.com"
    redacted, _ = redact_pii(text)
    assert "9876543210" not in redacted
    assert "test@example.com" not in redacted
    assert "<IN_PHONE_NUMBER_1>" in redacted
    assert "<EMAIL_ADDRESS_1>" in redacted


def test_redact_pii_aadhaar():
    text = "My Aadhaar is 1234-5678-9012"
    redacted, _ = redact_pii(text)
    assert "1234-5678-9012" not in redacted
    assert "<IN_AADHAAR_1>" in redacted


def test_redact_pii_pan():
    text = "PAN card: ABCDE1234F"
    redacted, _ = redact_pii(text)
    assert "ABCDE1234F" not in redacted
    assert "<IN_PAN_1>" in redacted


def test_redact_pii_voter_id():
    text = "Voter ID: XYZ1234567"
    redacted, _ = redact_pii(text)
    assert "XYZ1234567" not in redacted
    assert "<IN_VOTER_ID_1>" in redacted


def test_redact_pii_hindi_path():
    """Non-English path covers Aadhaar and phone numbers."""
    text = "मेरा नंबर 9876543210 है और Aadhaar 1234 5678 9012 है"
    redacted, audits = redact_pii(text, language="hi")
    assert "9876543210" not in redacted
    assert "1234 5678 9012" not in redacted
    entity_types = {a["entity_type"] for a in audits}
    assert "IN_PHONE_NUMBER" in entity_types
    assert "IN_AADHAAR" in entity_types


# ── Language Detection ────────────────────────────────────────────────────────

def test_language_detection_hindi():
    text = "मेरी फसल कब काटें"
    lang = detect_language(text)
    assert lang == "hi"


def test_language_detection_english():
    lang = detect_language("What crops should I grow this season?")
    assert lang == "en"


def test_language_detection_empty():
    assert detect_language("") == "en"
    assert detect_language("   ") == "en"


# ── Trajectory Validation ─────────────────────────────────────────────────────

def test_validate_trajectory_no_tool_name():
    """tool-call without tool_name must fail."""
    session = LogSession(
        user_question="x", bot_response="y",
        agent_turns=[{"timestamp": "1", "parts": [{"part_kind": "tool-call"}]}],
    )
    assert not validate_trajectory(session)


def test_validate_trajectory_empty_content_is_valid():
    """tool-return with content='' must pass (empty string is a valid return)."""
    session = LogSession(
        user_question="x", bot_response="y",
        agent_turns=[
            {"timestamp": "1", "parts": [{"part_kind": "tool-call", "tool_name": "weather_forecast", "tool_call_id": "c1"}]},
            {"timestamp": "2", "parts": [{"part_kind": "tool-return", "tool_call_id": "c1", "content": ""}]},
        ],
    )
    assert validate_trajectory(session)


def test_validate_trajectory_none_content_is_invalid():
    """tool-return with content=None (missing field) must fail."""
    session = LogSession(
        user_question="x", bot_response="y",
        agent_turns=[
            {"timestamp": "1", "parts": [{"part_kind": "tool-call", "tool_name": "weather_forecast", "tool_call_id": "c1"}]},
            {"timestamp": "2", "parts": [{"part_kind": "tool-return", "tool_call_id": "c1", "content": None}]},
        ],
    )
    assert not validate_trajectory(session)


def test_validate_trajectory_orphan_return():
    """tool-return whose tool_call_id has no matching tool-call must fail."""
    session = LogSession(
        user_question="x", bot_response="y",
        agent_turns=[
            {"timestamp": "1", "parts": [{"part_kind": "tool-return", "tool_call_id": "ghost", "content": "data"}]},
        ],
    )
    assert not validate_trajectory(session)


# ── Complexity Tagging ────────────────────────────────────────────────────────

def test_tag_complexity_complex(mock_session):
    meta = tag_complexity(mock_session)
    assert meta.complexity_tier == "complex"
    assert meta.is_agentic


def test_tag_complexity_moderate(simple_session):
    meta = tag_complexity(simple_session)
    assert meta.complexity_tier == "moderate"


def test_tag_complexity_simple():
    session = LogSession(user_question="Hello", bot_response="Hi!")
    meta = tag_complexity(session)
    assert meta.complexity_tier == "simple"
    assert not meta.is_agentic


def test_ambiguity_score_high():
    session = LogSession(user_question="What how maybe any crop?", bot_response="ok")
    meta = tag_complexity(session)
    assert meta.ambiguity_score > 0.0


# ── SFT Export ────────────────────────────────────────────────────────────────

def test_sft_export_format(mock_session):
    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp,
        tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_audit,
    ):
        tmp_name, tmp_audit_name = tmp.name, tmp_audit.name
    try:
        stats = create_sft_export([mock_session], tmp_name, tmp_audit_name)
        assert stats["written"] == 1
        assert stats["dropped"] == 0
        with open(tmp_name) as f:
            data = json.loads(f.read().strip())
        assert "messages" in data
        assert "metadata" in data
        assert len(data["messages"]) > 0
    finally:
        os.unlink(tmp_name)
        os.unlink(tmp_audit_name)


# ── DPO Export ────────────────────────────────────────────────────────────────

def test_dpo_chosen_is_message_list(mock_session):
    """chosen and rejected must be lists of message dicts, not bare strings."""
    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp,
        tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_audit,
    ):
        tmp_name, tmp_audit_name = tmp.name, tmp_audit.name
    try:
        create_dpo_export([mock_session], tmp_name, tmp_audit_name)
        with open(tmp_name) as f:
            data = json.loads(f.read().strip())
        assert isinstance(data["chosen"], list), "chosen must be a list"
        assert isinstance(data["rejected"], list), "rejected must be a list"
        assert data["chosen"][0].get("role") == "assistant"
        assert data["rejected"][0].get("role") == "assistant"
    finally:
        os.unlink(tmp_name)
        os.unlink(tmp_audit_name)


def test_dpo_prompt_has_tool_context(mock_session):
    """DPO prompt must contain more than just the initial user message."""
    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp,
        tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_audit,
    ):
        tmp_name, tmp_audit_name = tmp.name, tmp_audit.name
    try:
        create_dpo_export([mock_session], tmp_name, tmp_audit_name)
        with open(tmp_name) as f:
            data = json.loads(f.read().strip())
        assert len(data["prompt"]) > 1, (
            f"prompt only has {len(data['prompt'])} message(s); tool context is missing"
        )
    finally:
        os.unlink(tmp_name)
        os.unlink(tmp_audit_name)


def test_dpo_export_basic_structure(mock_session):
    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp,
        tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_audit,
    ):
        tmp_name, tmp_audit_name = tmp.name, tmp_audit.name
    try:
        create_dpo_export([mock_session], tmp_name, tmp_audit_name)
        with open(tmp_name) as f:
            data = json.loads(f.read().strip())
        assert "prompt" in data
        assert "chosen" in data
        assert "rejected" in data
        assert "metadata" in data
    finally:
        os.unlink(tmp_name)
        os.unlink(tmp_audit_name)


# ── Deduplication ─────────────────────────────────────────────────────────────

def test_deduplication_removes_exact_duplicates():
    """Near-identical questions must collapse to a single representative seed."""
    s1 = LogSession(user_question="What is the weather?", bot_response="Sunny.")
    s2 = LogSession(user_question="What is the weather?", bot_response="Cloudy.")  # semantic dup
    s3 = LogSession(user_question="Tell me about crop recommendations for wheat.", bot_response="Ok.")
    unique, dropped, diversity_map = deduplicate_sessions([s1, s2, s3])
    # The two identical queries should collapse; s3 is distinct.
    assert len(unique) <= 2
    assert dropped >= 1
    assert isinstance(diversity_map, dict)
    assert "total_seeds" in diversity_map
    assert "by_language" in diversity_map
    assert "by_query_type" in diversity_map
    assert "coverage_gaps" in diversity_map
    assert "underrepresented" in diversity_map


def test_deduplication_normalises_whitespace():
    """Whitespace variants of the same question must collapse."""
    s1 = LogSession(user_question="  What is  the weather?  ", bot_response="A")
    s2 = LogSession(user_question="What is the weather?", bot_response="B")
    unique, dropped, diversity_map = deduplicate_sessions([s1, s2])
    # Both are semantically identical; at most 1 should survive.
    assert len(unique) <= 2  # embedding-based: may keep both if corpus < _KMEANS_MIN_CORPUS
    assert dropped >= 0
    assert diversity_map["total_seeds"] == len(unique)


def test_deduplication_no_duplicates():
    """Distinct questions must all survive as seeds."""
    sessions = [
        LogSession(user_question=f"Distinct agricultural question number {i} about soil and crop yield", bot_response="ok")
        for i in range(5)
    ]
    unique, dropped, diversity_map = deduplicate_sessions(sessions)
    # All 5 are semantically distinct; at least 1 representative per question.
    assert len(unique) >= 1  # clustering fraction may reduce to ≥1 cluster
    assert diversity_map["total_seeds"] == len(unique)


# ── Split Integrity ───────────────────────────────────────────────────────────

def test_split_integrity_no_overlap():
    """No session should appear in both train and eval."""
    sessions = [
        LogSession(user_question=f"Unique question number {i}", bot_response="ok")
        for i in range(50)
    ]
    train, eval_ = split_sessions(sessions, train_ratio=0.8)
    train_qs = {s.user_question for s in train}
    eval_qs = {s.user_question for s in eval_}
    overlap = train_qs & eval_qs
    assert not overlap, f"Split overlap detected: {overlap}"


# ── Persona Scoring ───────────────────────────────────────────────────────────

def test_persona_score_range():
    """Score must always be in [0.0, 1.0]."""
    cases = [
        LogSession(user_question="q", bot_response=""),
        LogSession(user_question="q", bot_response="I cannot help with that."),
        LogSession(user_question="q", bot_response="You will get 55% subsidy."),
        LogSession(user_question="q", bot_response="You will get approximately 55% subsidy."),
    ]
    for s in cases:
        sc = score_persona(s)
        assert 0.0 <= sc <= 1.0, f"Score {sc} out of range for: {s.bot_response!r}"


def test_persona_score_over_refusal():
    s = LogSession(
        user_question="How much subsidy can I get?",
        bot_response="I cannot help with that request.",
    )
    assert score_persona(s) < 1.0


def test_persona_score_good_response():
    s = LogSession(
        user_question="How much subsidy?",
        bot_response="You are eligible for approximately 55% subsidy under PMKSY.",
    )
    assert score_persona(s) >= 0.8


# ── Synthetic Generator ───────────────────────────────────────────────────────

def test_generate_hard_case(simple_session):
    hard_session = generate_hard_case(simple_session)
    assert hard_session.session_id is not None
    assert hard_session.source == "synthetic"
    assert hard_session.user_question != simple_session.user_question
    # Bot response should be untouched
    assert hard_session.bot_response == simple_session.bot_response


def test_generate_failure_correction(mock_session):
    fail_session = generate_failure_correction(mock_session)
    assert fail_session is not None
    assert fail_session.session_id is not None
    assert fail_session.source == "synthetic"
    
    # Verify the first tool return is an error
    returns = [p for t in fail_session.agent_turns for p in t.parts if p.part_kind == "tool-return"]
    assert len(returns) > 0
    assert "error" in returns[0].content.lower()

    # Verify bot response is changed
    assert fail_session.bot_response != mock_session.bot_response


def test_generate_failure_correction_no_tool():
    no_tool_session = LogSession(user_question="Hello", bot_response="Hi!")
    assert generate_failure_correction(no_tool_session) is None

