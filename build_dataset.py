"""Export pipeline: SFT and DPO JSONL writers.

- Language is auto-detected before PII redaction so the correct code path
  always runs.
- Failed sessions are counted and reported, never silently dropped.
- Deduplication is applied before export.
- Persona scoring gates DPO chosen quality.
- DPO prompt includes full tool context (not just the first user message).
- DPO chosen / rejected are message lists, not bare strings.
- Synthetic negatives are semantically meaningful (three rejection types).
"""
import json
import os
import hashlib
import argparse
import collections
import logging

from lang_utils import ensure_language
from anonymizer import redact_pii
from types_def import LogSession, DPORecord
from analyzer import validate_trajectory, tag_complexity
from session_dedup import deduplicate_sessions
from behavior_scorer import score_persona
from data_augmenter import (
    generate_hard_case,
    generate_failure_correction,
)

logger = logging.getLogger(__name__)

# Sessions scoring below this threshold are excluded from DPO chosen
_PERSONA_THRESHOLD = 0.4


# ── Helpers ──────────────────────────────────────────────────────────────────

def split_sessions(
    sessions: list[LogSession], train_ratio: float = 0.8, seed: int = 42
) -> tuple[list[LogSession], list[LogSession]]:
    """Deterministic train/eval split based on a hash of user_question."""
    train_sessions: list[LogSession] = []
    eval_sessions: list[LogSession] = []
    for s in sessions:
        h = int(hashlib.md5(s.user_question.encode("utf-8")).hexdigest(), 16)
        if (h % 100) < (train_ratio * 100):
            train_sessions.append(s)
        else:
            eval_sessions.append(s)
    return train_sessions, eval_sessions


def get_trace_id(session: LogSession) -> str:
    key = session.session_id or session.user_question
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:12]


def _redact_session(
    session: LogSession,
) -> tuple[list[dict], list[dict]]:
    """Redact PII across all text fields and build the full message list.

    Returns
    -------
    (messages, audit_entries)
        ``messages`` follows the OpenAI chat format (user / assistant / tool).
    """
    # Auto-detect language if not already set
    ensure_language(session)

    counts: dict = collections.defaultdict(int)
    all_audit: list[dict] = []

    def r(text: str | None) -> str | None:
        if text is None:
            return None
        redacted, audits = redact_pii(text, language=session.language or "en", _counts=counts)
        all_audit.extend(audits)
        return redacted

    messages: list[dict] = [{"role": "user", "content": r(session.user_question)}]

    for turn in session.agent_turns:
        for part in turn.parts:
            if part.part_kind == "tool-call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": part.tool_name or "",
                                    "arguments": (
                                        json.dumps(
                                            {k: r(str(v)) for k, v in part.args.items()}
                                        )
                                        if part.args
                                        else "{}"
                                    ),
                                },
                                "id": part.tool_call_id or "",
                            }
                        ],
                    }
                )
            elif part.part_kind == "tool-return":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.tool_call_id or "",
                        "content": r(part.content),
                    }
                )

    messages.append({"role": "assistant", "content": r(session.bot_response)})
    return messages, all_audit


# ── Synthetic DPO rejection generation ───────────────────────────────────────

def _generate_rejection(
    messages: list[dict], session: LogSession
) -> tuple[list[dict], str]:
    """Build a *rejected* message list and return (rejected_messages, rejection_type).

    Three rejection strategies (chosen deterministically from trace_id):
    - ``persona_violation``: overconfident tone, drops hedges.
    - ``tool_skip``: hallucinates an answer without referencing tool results.
    - ``language_switch``: responds in English when session is non-English.
    """
    import hashlib

    strategy_idx = int(hashlib.md5(session.user_question.encode()).hexdigest(), 16) % 3
    original_response = messages[-1].get("content") or ""

    if strategy_idx == 0:
        # persona_violation — overly casual and unprofessional
        bad_response = (
            "Yeah sure, whatever. Here's your info: " + original_response.lstrip()
        )
        rejection_type = "persona_violation"

    elif strategy_idx == 1:
        # tool_skip — classic AI refusal (ignoring the fact it has tools)
        bad_response = (
            "I'm sorry, but as an AI language model, I don't have access to your personal records, "
            "Aadhaar details, or live weather data. I can only provide general information."
        )
        rejection_type = "tool_skip"

    else:
        # language_switch — responds in English for non-English sessions
        if session.language and session.language != "en":
            bad_response = (
                "I can only respond in English. Here is the answer: " + original_response
            )
            rejection_type = "language_switch"
        else:
            # Fall back to persona_violation for English sessions
            bad_response = "Absolutely: " + original_response
            rejection_type = "persona_violation"

    rejected_messages = [{"role": "assistant", "content": bad_response}]
    return rejected_messages, rejection_type


# ── Export functions ──────────────────────────────────────────────────────────

def create_sft_export(
    sessions: list[LogSession], output_file: str, audit_file_path: str
) -> dict:
    """Write SFT JSONL and return stats dict."""
    written = dropped = 0
    with open(output_file, "w", encoding="utf-8") as f, open(audit_file_path, "a", encoding="utf-8") as af:
        for session in sessions:
            if not validate_trajectory(session):
                logger.warning(
                    "SFT: dropping session with invalid trajectory (question: %r)",
                    session.user_question[:60],
                )
                dropped += 1
                continue

            messages, audit_entries = _redact_session(session)
            trace_id = get_trace_id(session)
            meta = tag_complexity(session)
            meta.trace_id = trace_id
            meta.persona_score = score_persona(session)

            if audit_entries:
                af.write(json.dumps({"trace_id": trace_id, "findings": audit_entries}, ensure_ascii=False) + "\n")

            f.write(json.dumps({"messages": messages, "metadata": meta.model_dump()}, ensure_ascii=False) + "\n")
            written += 1

    return {"written": written, "dropped": dropped}


def create_dpo_export(
    sessions: list[LogSession], output_file: str, audit_file_path: str
) -> dict:
    """Write DPO JSONL and return stats dict.

    DPO format
    ----------
    - ``prompt``: full conversation up to (not including) the final assistant turn.
    - ``chosen``: list containing the final assistant message (real response).
    - ``rejected``: list containing the synthetic negative response.

    Sessions are excluded when:
    - They have no agent turns (no tool context for DPO contrast).
    - Trajectory validation fails.
    - Persona score is below :data:`_PERSONA_THRESHOLD`.
    """
    written = dropped = 0
    with open(output_file, "w", encoding="utf-8") as f, open(audit_file_path, "a", encoding="utf-8") as af:
        for session in sessions:
            if not session.agent_turns:
                dropped += 1
                continue
            if not validate_trajectory(session):
                logger.warning(
                    "DPO: dropping session with invalid trajectory (question: %r)",
                    session.user_question[:60],
                )
                dropped += 1
                continue

            messages, audit_entries = _redact_session(session)
            trace_id = get_trace_id(session)

            if audit_entries:
                af.write(json.dumps({"trace_id": trace_id, "findings": audit_entries}, ensure_ascii=False) + "\n")

            # prompt = everything except the final assistant turn
            prompt = messages[:-1]
            # chosen = [final assistant message]
            chosen = [messages[-1]]
            # rejected = synthetic negative
            rejected, rejection_type = _generate_rejection(messages, session)

            meta = tag_complexity(session)
            meta.trace_id = trace_id
            meta.rejection_type = rejection_type
            meta.persona_score = score_persona(session)

            if meta.persona_score < _PERSONA_THRESHOLD:
                logger.warning(
                    "DPO: dropping session — persona score %.2f below threshold (question: %r)",
                    meta.persona_score,
                    session.user_question[:60],
                )
                dropped += 1
                continue

            dpo_record = DPORecord(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                metadata=meta,
                synthetic=True,
            )
            f.write(json.dumps(dpo_record.model_dump(), ensure_ascii=False) + "\n")
            written += 1

    return {"written": written, "dropped": dropped}


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _augment(sess_list: list[LogSession]) -> list[LogSession]:
    aug = []
    for s in sess_list:
        aug.append(generate_hard_case(s))
        if fail_s := generate_failure_correction(s):
            aug.append(fail_s)
    return sess_list + aug


def process_logs(input_file: str, output_dir: str, split_ratio: float) -> None:
    os.makedirs(output_dir, exist_ok=True)
    audit_file_path = os.path.join(output_dir, "audit_log.jsonl")
    if os.path.exists(audit_file_path):
        os.remove(audit_file_path)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sessions = [LogSession(**item) for item in data]

    # ── Semantic deduplication (embedding + FAISS clustering) ────────────────
    sessions, dup_count, diversity_map = deduplicate_sessions(sessions)
    print(
        f"Deduplication: {dup_count} near-duplicate session(s) collapsed into "
        f"{diversity_map['total_seeds']} seed(s)."
    )
    # --- Diversity map summary ---
    print("\nDiversity map (representative seeds):")
    print(f"  Total seeds      : {diversity_map['total_seeds']}")
    print(f"  By language      : {diversity_map['by_language']}")
    print(f"  By query type    : {diversity_map['by_query_type']}")
    if diversity_map["coverage_gaps"]:
        gaps = ", ".join(
            f"{g['language']}/{g['query_type']}" for g in diversity_map["coverage_gaps"]
        )
        print(f"  Coverage gaps    : {gaps}")
    else:
        print("  Coverage gaps    : none")
    if diversity_map["underrepresented"]:
        under = ", ".join(
            f"{u['language']}/{u['query_type']} ({u['count']} sample(s), "
            f"{u['share']*100:.1f}% vs {u['uniform_expected']*100:.1f}% expected)"
            for u in diversity_map["underrepresented"]
        )
        print(f"  Underrepresented : {under}")
    else:
        print("  Underrepresented : none")
    
    # Save the collapsed duplicates for auditing
    if "collapsed_groups" in diversity_map and diversity_map["collapsed_groups"]:
        audit_dedup_path = os.path.join(output_dir, "dedup_dropped.json")
        with open(audit_dedup_path, "w", encoding="utf-8") as f:
            json.dump(diversity_map["collapsed_groups"], f, indent=4, ensure_ascii=False)
        print(f"  [Auditing] Saved collapsed duplicates to: {audit_dedup_path}")
    print()

    train_sessions, eval_sessions = split_sessions(sessions, train_ratio=split_ratio)

    train_sessions = _augment(train_sessions)
    eval_sessions = _augment(eval_sessions)
    print(f"Synthetic generation: augmented datasets to {len(train_sessions)} train, {len(eval_sessions)} eval.")

    sft_train_stats = create_sft_export(
        train_sessions, os.path.join(output_dir, "sft_train.jsonl"), audit_file_path
    )
    sft_eval_stats = create_sft_export(
        eval_sessions, os.path.join(output_dir, "sft_eval.jsonl"), audit_file_path
    )
    dpo_train_stats = create_dpo_export(
        train_sessions, os.path.join(output_dir, "dpo_train.jsonl"), audit_file_path
    )
    dpo_eval_stats = create_dpo_export(
        eval_sessions, os.path.join(output_dir, "dpo_eval.jsonl"), audit_file_path
    )

    print(
        f"Processed {len(sessions)} unique sessions → {output_dir}\n"
        f"  SFT  train: {sft_train_stats['written']} written, "
        f"{sft_train_stats['dropped']} dropped\n"
        f"  SFT  eval : {sft_eval_stats['written']} written, "
        f"{sft_eval_stats['dropped']} dropped\n"
        f"  DPO  train: {dpo_train_stats['written']} written, "
        f"{dpo_train_stats['dropped']} dropped\n"
        f"  DPO  eval : {dpo_eval_stats['written']} written, "
        f"{dpo_eval_stats['dropped']} dropped"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="user_interactions.json")
    parser.add_argument("--output", default="output/")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    args = parser.parse_args()

    if os.path.exists(args.input):
        process_logs(args.input, args.output, args.split_ratio)