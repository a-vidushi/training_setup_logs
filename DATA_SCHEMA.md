# Schema & Pipeline Documentation

## Input Log Schema (`LogSession`)

| Field | Type | Description |
|---|---|---|
| `user_question` | string | Raw user query (PII-redacted before export) |
| `bot_response` | string | Final model response (PII-redacted before export) |
| `agent_turns` | list | Ordered tool-call / tool-return turns between question and response |
| `session_id` | string? | Optional session identifier (not exported to training artifacts) |
| `language` | string? | Optional language tag (e.g. `en`, `hi`) |
| `domain` | string? | Optional domain tag (e.g. `agri`, `weather`) |

### `AgentTurn.parts[].part_kind` values

| Value | Meaning |
|---|---|
| `tool-call` | Model requested a tool; has `tool_name`, `args`, `tool_call_id` |
| `tool-return` | Tool responded; has `tool_name`, `content`, `tool_call_id` |

---

## SFT JSONL Schema (`sft_train.jsonl`, `sft_eval.jsonl`)

Each line is a JSON object:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [{"type": "function", "function": {"name": "...", "arguments": "{}"}, "id": "call_1"}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "..."},
    {"role": "assistant", "content": "final response"}
  ],
  "metadata": {
    "tool_count": 2,
    "unique_tools": ["weather_forecast", "fetch_agristack_data"],
    "has_recovery": false,
    "complexity_tier": "moderate",
    "is_agentic": true
  }
}
```

- Compatible with TRL `SFTTrainer` and Hugging Face `datasets` directly.
- Tool-call format matches OpenAI function-calling chat template used by Gemma, Llama, Qwen.
- All `content` fields are PII-redacted before writing.

---

## DPO JSONL Schema (`dpo_train.jsonl`, `dpo_eval.jsonl`)

Each line is a JSON object:

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "correct model response"}],
  "rejected": [{"role": "assistant", "content": "I cannot help with that. correct model response"}],
  "metadata": { "...same as SFT metadata..." },
  "synthetic": true
}
```

- `prompt`: conversation prefix up to and including the last user turn.
- `chosen`: PII-redacted actual bot response.
- `rejected`: synthetic negative (prefixed refusal). `"synthetic": true` marks these for filtering once real preference pairs (thumbs, edits, failure/success pairs) are available.
- Compatible with TRL `DPOTrainer` (`prompt`, `chosen`, `rejected` are the required fields).

---

## Complexity Tags & Recommended Training Schedule

| `complexity_tier` | Condition | Recommended use |
|---|---|---|
| `simple` | 0 tool calls | Warm-up / instruction-following phase |
| `moderate` | 1–2 tool calls, no recovery | Main SFT bulk training |
| `complex` | 3+ tool calls **or** `has_recovery=true` | Later-stage fine-tuning; up-weight in DPO |

Trainers can filter or sample by `metadata.complexity_tier` for staged curriculum or flat mixture runs without any pipeline changes.

---

## Split Strategy

Sessions are assigned to `train` or `eval` using a **hash of `user_question`** (MD5 mod 100), not a random index. This ensures:
- Near-duplicate or repeated queries always land in the **same split**.
- No prompt leakage between `sft_train` and `dpo_train` preference sets.
- Splits are **deterministic and reproducible** across pipeline runs.

Default ratio: 80% train / 20% eval (configurable via `--split-ratio`).

---

## PII Handling

Detected entity types: `PHONE_NUMBER`, `EMAIL_ADDRESS`, `URL`, `IN_AADHAAR`

- Replaced with indexed placeholders: `<PHONE_NUMBER_1>`, `<EMAIL_ADDRESS_1>`, `<IN_AADHAAR_1>`, etc.
- Placeholders are **consistent within a session** (same entity → same placeholder across SFT and DPO exports for that session).
- Raw PII is logged to an in-memory audit list accessible via `audit_sample()` in `anonymizer.py` for human review of false negatives.
- `PERSON` entity detection is intentionally disabled to avoid false positives on crop names and place names in agricultural context.

**Residual risk:** Indirect identification via rare location + crop combinations is not eliminated by redaction alone. Human audit sampling is required before production export.

---

## Student Model Criteria

Target: smallest dense model ≤ 32B parameters that achieves parity with the teacher on the held-out `sft_eval.jsonl` set.

Acceptance threshold (to be confirmed with mentor):
- Tool-call accuracy (correct tool name + valid args) ≥ teacher baseline on `sft_eval`
- Response quality score (persona adherence, checked via rule layer for verifiable rules) within 5% of teacher
- End-to-end latency: 4–5 tool calls + final response within ~4 seconds on 8×H100

Filter applied to student training data: exclude sessions where `tool_count > 4` or total token length exceeds the student model's context window.