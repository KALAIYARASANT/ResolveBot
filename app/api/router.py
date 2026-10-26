import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field

from fastapi import APIRouter, Header, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.db.database import customer_exists
from app.schemas.chat import ChatRequest, ChatResponse
from app.agents.support_agent import is_clear_confirmation, run_chat
from langchain_core.messages import AIMessage, HumanMessage

router = APIRouter()
REQUEST_TIMEOUT_SECONDS = 27.0


@dataclass
class Session:
    customer_id: str
    history: list = field(default_factory=list)
    pending: dict | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    confirmed_action_reply: str | None = None


SESSIONS: dict[str, Session] = {}


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, x_customer_id: str | None = Header(default=None, alias="X-Customer-Id")):
    deadline = time.monotonic() + 30.0
    request_id = str(uuid.uuid4())
    if not x_customer_id or not customer_exists(x_customer_id):
        return JSONResponse(status_code=401, content={"detail": "Missing or unknown customer ID"})
    try:
        payload = await request.json()
        body = ChatRequest.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON or missing/invalid required request fields"})
    session = SESSIONS.get(body.session_id)
    if session and session.customer_id != x_customer_id:
        return JSONResponse(status_code=403, content={"detail": "Session belongs to a different customer"})
    if not session:
        session = SESSIONS[body.session_id] = Session(customer_id=x_customer_id)
    async with session.lock:
        previous_pending = session.pending
        was_confirmation = bool(previous_pending and is_clear_confirmation(body.message))
        if not previous_pending and session.confirmed_action_reply and is_clear_confirmation(body.message):
            answer, route, calls, citations, pending, new_proposal = (
                session.confirmed_action_reply, "answer", [], [], False, None
            )
        elif was_confirmation:
            # Confirmation is handled synchronously: a timeout must never leave a
            # background thread that can insert after the caller receives a timeout.
            answer, route, calls, citations, pending, new_proposal = run_chat(
                x_customer_id, session.history, body.message, previous_pending
            )
        else:
            # Any non-confirmation is the immediate next message and expires the
            # proposal before normal processing resumes.
            session.pending = None
            try:
                remaining = max(0.0, deadline - time.monotonic())
                timeout = min(REQUEST_TIMEOUT_SECONDS, remaining)
                if timeout <= 0:
                    raise asyncio.TimeoutError
                answer, route, calls, citations, pending, new_proposal = await asyncio.wait_for(
                    asyncio.to_thread(run_chat, x_customer_id, session.history, body.message, previous_pending),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                answer = "I couldn't complete that request within 30 seconds. Please try again."
                route, calls, citations, pending, new_proposal = "no_answer", [], [], False, None
        if route == "out_of_scope":
            calls, citations, pending, new_proposal = [], [], False, None
        if was_confirmation and any(call.get("status") == "success" for call in calls):
            session.confirmed_action_reply = answer
        elif not (session.confirmed_action_reply and is_clear_confirmation(body.message)):
            session.confirmed_action_reply = None
        session.pending = new_proposal
        session.history.extend([HumanMessage(content=body.message), AIMessage(content=answer)])
    return {"request_id": request_id, "answer": answer, "route": route, "citations": citations,
            "tool_calls": calls, "pending_confirmation": pending}
