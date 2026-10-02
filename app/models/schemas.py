"""Request and response schemas for financial filing review."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, pattern=r"\S", examples=["Review filing_001 with excerpt references."])
    session_id: str | None = None
    provider: Literal["gemini"] = "gemini"
    model: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=1.0)
    api_key: str | None = None
    agent_type: Literal["tool_calling"] = "tool_calling"


class ToolCallStep(BaseModel):
    tool: str
    tool_input: Any
    tool_output: str


class QueryResponse(BaseModel):
    query: str
    response: str
    session_id: str
    provider_used: str
    model_used: str
    steps: list[ToolCallStep] = Field(default_factory=list)
    execution_time_seconds: float


class ChatMessage(BaseModel):
    role: str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    message_count: int
    messages: list[ChatMessage]


class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    default_provider: str
