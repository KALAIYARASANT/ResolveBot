"""Customer-facing wording: Python decides, the LLM writes.

Deterministic turns (return checks, confirmations, clarifications) hand back a facts dict instead of a sentence.
compose_reply has the model turn those facts into the reply. Facts whose keys start with "_" are for Python
only: "_context" is stored with the turn so the next message can be understood, "_must" lists strings the
reply has to contain (a reference ID, the fixed confirmation question).
"""
import json
import logging
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings

COMPOSE_SYSTEM = " ".join([
    "You are ResolveBot, ShopSphere's customer support assistant.",
    "Python has already made every decision for this turn: what is eligible, what was saved, what is still needed.",
    "Write the reply to the customer's latest message using ONLY the FACTS in the last message.",
    "Do not change, doubt or add to them: no extra policies, dates, numbers, IDs or promises.",
    "Whenever the facts include `item`, name it in the reply with its order and item IDs, even when you only ask a question.",
    "If `can_be_returned` or `eligible_for_return` is present, begin by saying plainly whether the item can be returned (naming the item), then why: how many days ago it was delivered and what the return window is, or the `detail` if it cannot.",
    "When `reason` is given, say it in plain words (changed their mind, or defective). For a prepared request, say what you would submit: the item and the reason.",
    "If only one reason qualifies, say which one and why the other does not.",
    "If `ask` is present, end by asking the customer for exactly those things, in one short question.",
    "If `end_with` is present, finish the reply with exactly that sentence.",
    "Include `reference_id` when it is present.",
    "If the facts say nothing was saved or submitted, never imply that it was.",
    "Never mention tools, facts, JSON, Python or 'the system', and never reveal another customer's data.",
    "Plain text without markdown, friendly and concise (one to three sentences). Write dates like 3 July 2026.",
])


OUTAGE_NOTICE = "Sorry, I couldn't put a full reply together just now. Please try again in a moment."


def adapt_to_model_error(exc: Exception, settings) -> bool:
    """A reasoning model (for example gpt-6-luna) rejects temperature=0 unless reasoning is switched off.

    Learn that once from the model's own error and carry on, so a missing OPENAI_REASONING_EFFORT does not
    break every reply. Returns True when the setting was changed and the call is worth repeating.
    """
    if "temperature" in str(exc).casefold() and not settings.openai_reasoning_effort:
        settings.openai_reasoning_effort = "none"
        logging.getLogger(__name__).warning("model rejected temperature=0; continuing with reasoning_effort=none")
        return True
    return False


class Reply(str):
    """Reply text that also carries the facts it was written from and the context the next turn needs."""
    facts: dict
    context: dict | None


def _public(facts: dict) -> dict:
    return {key: value for key, value in facts.items() if not key.startswith("_")}


def _plain(public: dict) -> str:
    """No model available: state the facts unembellished."""
    return "; ".join(f"{key}: {value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}"
                     for key, value in public.items())


def _text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(block["text"] for block in content if isinstance(block, dict) and isinstance(block.get("text"), str)).strip()
    return str(content).strip()


def compose_reply(history: list, message: str, facts: dict, chat_class) -> Reply:
    """Turn a facts dict into the customer's reply; chat_class is the chat model class (ChatOpenAI)."""
    public, settings = _public(facts), get_settings()
    text = ""
    if settings.openai_api_key and settings.llm_provider == "openai":
        today = date.fromisoformat(settings.app_today)
        prompt = [
            SystemMessage(content=f"{COMPOSE_SYSTEM} Today's date is {today.day} {today:%B %Y}."),
            *history[-6:], HumanMessage(content=message),
            SystemMessage(content="FACTS (JSON):\n" + json.dumps(public, ensure_ascii=False, default=str)),
        ]
        for _attempt in range(3):  # a slow call, or one that teaches us a setting, is retried rather than failing the customer
            extra = {"reasoning_effort": settings.openai_reasoning_effort} if settings.openai_reasoning_effort else {}
            try:
                model = chat_class(model=settings.openai_model, temperature=0, api_key=settings.openai_api_key,
                                   timeout=11, max_retries=0, **extra)
                text = _text(model.invoke(prompt).content).replace("**", "")
            except Exception as exc:
                logging.getLogger(__name__).warning("reply composition failed (%s: %s)", type(exc).__name__, str(exc)[:200])
                text = ""
                adapt_to_model_error(exc, settings)
            if text:
                break
        # The model was configured but could not be reached: apologise, never show the customer the internal facts.
        text = text or OUTAGE_NOTICE
    text = text or _plain(public)  # no API key at all: the plain facts
    for required in facts.get("_must", []):
        if required.casefold() not in text.casefold():
            text = f"{text} {required}".strip()
    end = facts.get("end_with")
    if end and not text.rstrip().endswith(end):
        text = f"{text.rstrip()} {end}"
    reply = Reply(text)
    reply.facts, reply.context = public, facts.get("_context")
    return reply
