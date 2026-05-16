from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Part(BaseModel):
    part_kind: str
    tool_name: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    content: Optional[str] = None
    tool_call_id: Optional[str] = None


class AgentTurn(BaseModel):
    parts: List[Part]
    timestamp: str
    tool_call_id: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    finish_reason: Optional[str] = None
    run_id: Optional[str] = None
    provider_name: Optional[str] = None


class LogSession(BaseModel):
    session_id: Optional[str] = None
    language: Optional[str] = None
    domain: Optional[str] = None
    # source: "production" | "synthetic" | "corrected"
    source: Optional[str] = "production"
    user_question: str
    bot_response: str
    agent_turns: List[AgentTurn] = Field(default_factory=list)


class TrainingMetadata(BaseModel):
    tool_count: int
    unique_tools: List[str]
    has_recovery: bool
    complexity_tier: str
    is_agentic: bool
    total_tokens: int = 0
    student_eligible: bool = True
    trace_id: Optional[str] = None
    rejection_type: Optional[str] = None
    # Multi-turn depth: number of distinct user turns (≥1 for all sessions)
    multi_turn_depth: int = 1
    # Ambiguity score: 0.0 (unambiguous) – 1.0 (highly ambiguous question)
    ambiguity_score: float = 0.0
    # Persona adherence score: 0.0 – 1.0 (set by persona_scorer)
    persona_score: float = 1.0


class DPORecord(BaseModel):
    # prompt: full conversation up to (not including) the final assistant turn
    prompt: List[Dict[str, Any]]
    # chosen / rejected are message lists, not bare strings (TRL DPOTrainer compatible)
    chosen: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]
    metadata: TrainingMetadata
    synthetic: Optional[bool] = None