from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


class ToolCallRecord(BaseModel):
    name: str
    args: dict
    status: str


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    route: Literal["answer", "clarify", "no_answer", "out_of_scope"]
    citations: list[dict] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    pending_confirmation: bool = False
