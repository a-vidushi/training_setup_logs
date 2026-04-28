# Problem Statement: From Production Logs to Safe, SOTA-Aligned Training Data

This document describes the **problem space** for turning real usage logs into high-quality training data—after privacy-safe processing—so that fine-tuned or post-trained models improve on both **question answering** and **agentic** behavior.

---

## One-line goal

**Ingest logs → remove PII → build a trajectory-oriented pipeline** (controlled diversity plus explicit **trajectory-complexity scheduling**) **that exports SFT (LoRA) and DPO–ready data**, improves behavior, and supports training a **smaller replacement model**—all without leaking sensitive information.

---

## Context

- **Sources:** Application logs that mix **direct Q&A** (user question, model answer, possibly retrieval context) with **agentic sessions** (multi-turn dialogue, tool or API calls, environment feedback, partial failures, retries, and setup/configuration steps).
- **Constraint:** Logs contain **personally identifiable information (PII)** and possibly sensitive business data. Nothing that identifies individuals or confidential entities should reach training stores or third-party trainers without explicit policy, consent, and technical controls.
- **Outcome:** A **repeatable pipeline** that produces datasets suitable for **state-of-the-art (SOTA) practice**: not merely “pairs of strings,” but **trajectory-shaped** supervision where needed, distributions and splits, **tags for compositional complexity** (steps, tools, recovery), diversity guarantees, and—where relevant—**verifiable multi-turn traces** aligned with how capable agents are trained and evaluated. Exported data should be usable **without rework** in common **SFT (LoRA)** and **DPO** stacks (e.g. Hugging Face `datasets` + PEFT / TRL-style trainers), and structured so a **student** model can later match the **teacher** behavior seen in production.

This repository (or project folder) is a **staging ground** for defining requirements, experiments, and eventually implementations for that pipeline—not necessarily the final training code itself.


## Core sub-problems

### 1. Log understanding and segmentation

- **Normalize** heterogeneous log formats (JSON, plain text, tracing spans) into a common **event schema** (user turn, assistant turn, tool invocation, tool result, system message, errors).
- **Segment** continuous streams into **training units**: single-turn Q&A, multi-turn threads, and **agent trajectories** (tool graphs, not only final answers).
- **Label or infer metadata** where possible: task type, domain, language, success/failure, latency, tool names, number of hops—without re-identifying users.

### 2. PII removal and privacy-preserving use

- **Discover** PII (names, emails, phone numbers, addresses, IDs, URLs with tokens, free-text secrets) using rules, NER, and organization-specific dictionaries.
- **Redact or replace** with consistent placeholders (e.g. synthetic tokens) so that **statistics and linguistic patterns** remain useful while **identifiers** do not.
- **Audit:** sampling and human review for false negatives; document what was removed and residual risk (e.g. rare indirect identification).
- Align with **data retention, purpose limitation, and subprocessors** policies before any export for training.

### 3. From cleaned logs to training examples

Depending on the training stage, targets may include:

| Stage / objective | Typical data derived from logs |
|-------------------|--------------------------------|
| **SFT (LoRA) — Q&A** | Instruction → response, or short multi-turn chat in the same template as inference. |
| **SFT (LoRA) — agentic** | **Full or segmented trajectories** in chat/tool format: state → action (tool) → observation → next action; behavior-cloning targets from successful (or corrected) rollouts. |
| **Preference / DPO** | Shared prefix plus **chosen vs. rejected** suffixes: edits, thumbs, failed vs. successful trace for same intent, or governed synthetic negatives. |
| **Evaluation** | Held-out sets mirroring production **mix** by **trajectory complexity** and tool mix, not only loss on text. |

### Training-method compatibility: SFT (LoRA) and DPO

The pipeline should treat **export formats** as first-class requirements so the same underlying examples (or parallel views) can feed **supervised fine-tuning with LoRA** and **Direct Preference Optimization** without ad-hoc rewrites.

**SFT (LoRA):**

- Emit **standard chat or completion records** (e.g. `messages` arrays or prompt/completion pairs) that match **inference-time templates** exactly: same system prompt structure, tool-call syntax, stop tokens, and tokenizer chat template as the trainer requires
- **Multi-turn** agent rows should be valid **full trajectories** in one training example where the trainer expects that—so LoRA updates see tool boundaries and user/assistant alternation consistently.
- LoRA is a **parameter-efficiency** choice at train time; the dataset itself is still **standard SFT JSON/JSONL**—the pipeline documents one canonical schema and optional shims for specific libraries.

**DPO:**

- Each preference row needs a **shared prompt** (conversation prefix up to the last user turn, or equivalent) plus **chosen** and **rejected** completions—full strings or message suffixes as required by the DPO implementation 
- **Sources of pairs** from logs: explicit thumbs up/down, edits that replace model text, successful vs. failed trajectories for the same intent, or **synthetic** rejections (policy-violating or tool-wrong) generated under governance—always with **PII already stripped** from both branches.
- **Balance and length:** Chosen/rejected should be comparable where possible (avoid trivial “reject = empty string” unless that is the real product behavior) so the reward model implicit in DPO is not dominated by length or formatting artifacts.

**Shared constraints for both methods:**

- Same **tokenization and chat template** assumptions across SFT and DPO exports for a given training run.
- **Splits** that prevent leakage of near-identical prompts across SFT pretrain mix and DPO preference sets if both are built from the same logs.


---

## Success criteria (what “done” looks like for the problem, not one script)

1. **Privacy:** Documented PII handling; measurable redaction quality; no training on raw secrets.
2. **Coverage:** Explicit diversity and **trajectory-complexity targets** and reporting (not only row counts).
3. **Scheduling-ready:** Examples or shards tagged so trainers can run **flat mixtures**, **staged phases**, or **model-aware** sampling over complexity—not a single brittle static sort.
4. **Agent-ready:** Tool-use trajectories validate against schemas and, where possible, **execution** or outcome checks.
5. **Evaluation:** Held-out and behavioral metrics show improvement without regressions on safety or format.
6. **Training exports:** Documented, validated **SFT (LoRA-ready)** and **DPO-ready** dataset schemas (including multi-turn agent rows where applicable).
7. **Student path:** Clear subset of data and eval criteria aimed at **matching teacher behavior** with a **smaller** deployable model in the same setup.


## Next steps (for implementers)

1. Define the **canonical log schema** and **PII taxonomy** for your product.
2. Prototype **redaction** and measure **residual PII** on a sample.
3. Build a **small gold set** (human-curated) for Q&A and for **one** representative agent workflow; align all automated extraction to match that quality bar.
4. Add **tagging** (compositional complexity, domain, tools, outcome) and report **diversity** and **trajectory-scheduling** coverage before scaling volume.
5. Specify **one SFT JSONL schema** and **one DPO JSONL schema** (and chat template) validated end-to-end with a **LoRA** dry run and a **small DPO** dry run on toy data.
6. Define **student model** constraints (context length, tool set) and a **filter + eval** plan for teacher-to-student parity before production swap.


