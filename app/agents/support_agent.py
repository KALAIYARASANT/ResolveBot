import re
from datetime import date
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages

from app.core.config import get_settings
from app.core.dates import add_default_year, find_date
from app.agents.replies import Reply, adapt_to_model_error, compose_reply
from app.core.ids import normalize_order_ids
from app.db.database import connect, owned_order, policy_parameter
from app.services.business_rules import TODAY, validate_ticket
from app.services.policy_search import search_policy, significant_terms
from app.tools.support_tools import build_tools, execute_proposal

MAX_TOOL_CALLS = 6
_CONFIRMATIONS = {"yes", "yeah", "yep", "sure", "okay", "ok", "absolutely", "confirm", "confirmed", "go ahead", "please do", "yes go ahead", "yes please", "i confirm", "please proceed", "that s right"}
_ACTION_REQUEST = re.compile(r"\b(return(?:s|ing)?|cancel|cancellation|create|open|submit|file|request|ticket|claim)\b", re.I)
_DETAIL_CHANGE = re.compile(r"^(?:(?:actually[,]?\s+)?(?:just|only)\s+(?:the\s+)?|(?:actually[,]?\s+)?(?:change it to|switch (?:it )?to|replace (?:it )?(?:with|by)|make it)\s+(?:the\s+)?)[a-z0-9][a-z0-9 -]{1,50}$", re.I)
_ORDER_LOOKUP_INTENT = re.compile(r"\b(?:where(?:'s| is)|track(?:ing)?|status of|delivery (?:date|status)|eta|when (?:will|does) .{0,45} arrive|has .{0,45} shipped)\b", re.I)
_RETURN_BROAD_REFERENCE = re.compile(r"\b(?:my\s+(?:products?|items?)|all\s+(?:my\s+)?(?:products?|items?)|which\s+of\s+my\s+(?:products?|items?)|these\s+(?:products?|items?)|this\s+(?:item|product)|that\s+(?:item|product))\b", re.I)
_RETURN_OFFER = re.compile(r"\b(?:would you like me to|shall i|should i)\b.{0,100}\b(?:return|send.{0,12}back)\b", re.I | re.S)
_REQUESTS_DETAILS = re.compile(r"\b(?:please (?:share|provide|tell|confirm|let)|let me know)\b", re.I)
_RETURN_CLARIFICATION_CONTEXT = re.compile(r"\b(?:return|send.{0,12}back)\b.{0,160}\b(?:which|what|share|provide|order|item|product|reason|eligible|eligibility)\b", re.I | re.S)


def is_clear_confirmation(message: str) -> bool:
    normalized = " ".join(re.findall(r"[a-z]+", message.casefold()))
    if normalized in _CONFIRMATIONS:
        return True
    return bool(re.fullmatch(
        r"(?:yes|yeah|yep|sure|okay|ok|absolutely|confirm|confirmed) "
        r"(?:please )?(?:return(?: (?:request|it))?|go ahead|proceed|submit(?: it)?|"
        r"create (?:it|the return)|start (?:it|the return))(?: please)?",
        normalized,
    ))


def is_clear_decline(message: str) -> bool:
    normalized = " ".join(re.findall(r"[a-z]+", message.casefold()))
    return normalized in {"no", "nope", "no thanks", "no thank you", "not now", "cancel", "cancel it", "dont", "do not"}


def allows_new_proposal(message: str, previous_proposal: dict | None) -> bool:
    text = message.strip()
    if is_clear_confirmation(text) or re.fullmatch(r"(?:just\s+)?do it[.!\s]*", text, re.I):
        return False
    # A question about an action ("Can I return it? It's defective.") is a question even when more text follows.
    if re.search(r"\?\s*$", text) or any(_ACTION_REQUEST.search(q) for q in re.findall(r"[^.!?]*\?", text)):
        explicit_request = re.match(r"\s*(?:can|could|would)\s+you\s+(?:please\s+)?(?:return|cancel|create|open|submit|file|request|ticket|claim)\b", text, re.I)
        if not explicit_request and not re.search(r"\b(?:please|i want to|i'd like to)\b", text, re.I):
            return False
    if _ACTION_REQUEST.search(text):
        return True
    changed_details = text.rstrip(".,!;")
    return bool(previous_proposal and _DETAIL_CHANGE.fullmatch(changed_details))


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    tool_calls: list[dict]
    citations: list[dict]
    excerpts: list[dict]
    proposal: dict | None
    customer_id: str
    tools: list[Any]
    call_count: int
    policy_uncovered: bool
    policy_searched: bool
    policy_covered: bool
    order_found: bool
    non_policy_tool_used: bool
    final_answer: str
    order_not_found: bool
    tool_limit_reached: bool
    warranty_orders: list[str]


_CONFIRM_QUESTION = "Shall I go ahead?"
_ORDER_NOT_FOUND_ANSWER = "I couldn't find that order on your account."
_UNAVAILABLE_REPLY = "I couldn't complete that request right now. Please try again or contact ShopSphere support."

SYSTEM = " ".join([
    "You are ResolveBot, ShopSphere customer support. Use tools for order facts and policy questions. Never invent facts, policies, IDs, or eligibility.",
    "Return eligibility is determined only by the Python tool. Never ask for or accept a customer ID; it comes from trusted request context.",
    "Customers do not need to know an order ID: use get_orders to find their own products when appropriate.",
    "If an item or order is missing or several products/orders match, list the relevant account choices and ask which one; ask for every missing eligibility detail, including the reason, together. Do not check every account item when a broad question has not identified an item.",
    "If one message asks several things, answer each one separately, using the right tool and the order ID named in that part of the message.",
    "When the customer says 'it' or 'that order', use the single order discussed in the previous turn; if several were discussed, ask which one.",
    "In scope: the customer's own ShopSphere orders, and anything about how ShopSphere itself works (shipping and delivery, fees, payments, returns, refunds, warranty, cancellations, address changes, membership, what it sells). For these, ALWAYS call get_orders and/or search_policy before answering; never answer from memory. For a policy-only question (fees, thresholds, how a policy works) that does not concern one of the customer's own orders, call only search_policy and do not call get_orders.",
    "Out of scope: anything not about ShopSphere (coding help, general knowledge, sports, weather, news, opinions, creative writing). For those call no tools, say politely in one line that you can only help with ShopSphere orders and store policies (tracking, returns, warranty, cancellations, shipping), and make the last line `Route: out_of_scope`.",
    "Policy answers: rely only on search_policy excerpts. A policy document outranks the FAQ if they differ; if two current excerpts give different values for the same thing, the one with the later effective_date applies.",
    "Only after search_policy has returned nothing that answers an in-scope question, say that ShopSphere's policies do not cover it and suggest emailing the support team; never guess.",
    "Finish every reply with a last line `Route: answer`, or `Route: clarify` when you need more details from the customer first, or `Route: out_of_scope`. This line is removed before the customer sees the reply.",
    "Do not put document IDs or section numbers in the reply text. Instead, whenever you called search_policy, finish with one last line `Sources: <doc_id> <section>, <doc_id> <section>` naming only the excerpts you actually relied on, copied exactly from the excerpts, or `Sources: none`.",
    "If the customer mentions a date or a future change, include that date in the search_policy query. If an excerpt is marked as a future policy, state its effective date and distinguish it from current policy.",
    "Warranty questions: call get_orders and pick the item the customer means from each item's name and category (ask which one only if several items fit); take warranty_active and warranty_end_date from it, and call search_policy to see whether the fault is covered.",
    "If the item is under warranty and the fault is covered, say so and offer to open a warranty ticket, but do not propose one until the customer asks. If the warranty has ended or the product has none, say so and do not offer a ticket.",
    "For a possible write, propose only when the customer clearly requests the action; questions like 'Can I return this?' ask for eligibility only.",
    f"A proposal is not an execution: once create_return_request or create_support_ticket has prepared one, describe what you will do and end the reply with exactly \"{_CONFIRM_QUESTION}\". Use that phrase only then; to merely offer an action, ask \"Would you like me to ...?\".",
    "Tool writes are gated by Python and confirmation. Customer dates without a year refer to 2026; write dates like 3 July 2026. Keep answers concise and customer-friendly.",
])


def _is_compound(message: str) -> bool:
    """More than one thing asked (several orders, tracking plus return, or several questions): leave it to the agent loop.

    The scripted return shortcuts handle one return request at a time and would answer only the first order named.
    """
    order_ids = {m.upper() for m in re.findall(r"\bSS-\d+\b", message, re.I)}
    wants_return = bool(re.search(r"\b(?:return|send.{0,12}back)\b", message, re.I))
    return len(order_ids) > 1 or message.count("?") > 1 or (wants_return and bool(_ORDER_LOOKUP_INTENT.search(message)))


_SOURCES_LINE = re.compile(r"^[ \t>*_]*sources?\s*:\s*(?P<list>.*?)[ \t*_]*$", re.I | re.M)
_SOURCE_CONNECTORS = {"and", "or", "also", "plus", "with", "see", "none"}
_SOURCE_TOKEN = re.compile(r"(?P<doc>[a-z][a-z0-9_]*)|§\s*(?P<section>\d+(?:\.\d+)*)", re.I)


_ROUTE_LINE = re.compile(r"^[ \t>*_]*route\s*:\s*(?P<route>[a-z_]+)[ \t*_]*$", re.I | re.M)


def _split_route(answer: str):
    """Remove the model's closing `Route:` line; return the text and the route it named (None if it gave none)."""
    matches = list(_ROUTE_LINE.finditer(answer))
    if not matches:
        return answer, None
    last = matches[-1]
    return (answer[:last.start()] + answer[last.end():]).strip(), last.group("route").casefold()


def _split_sources(answer: str):
    """Remove the model's closing `Sources:` line; return the text and the (doc_id, section) pairs it named.

    The pairs are None when the model gave no line and an empty list for `Sources: none`.
    """
    matches = list(_SOURCES_LINE.finditer(answer))
    if not matches:
        return answer, None
    last = matches[-1]
    pairs, doc = [], None
    for token in _SOURCE_TOKEN.finditer(last.group("list")):
        if token.group("doc"):
            if token.group("doc").casefold() not in _SOURCE_CONNECTORS:
                doc = token.group("doc")
        elif doc:
            pairs.append((doc, "§" + token.group("section")))
    return (answer[:last.start()] + answer[last.end():]).strip(), pairs


MIN_NAMED_OVERLAP = 2   # shared significant terms for a section the model names to count as used
MIN_ANSWER_OVERLAP = 3  # shared significant terms for an unnamed excerpt to be cited as the answer's source


def _choose_citations(named, candidates: list[dict], excerpts: list[dict], answer: str) -> list[dict]:
    """Cite only retrieved excerpts that were used.

    That is the excerpts the model named (if retrieved and visible in its answer) plus the one excerpt the
    answer is most clearly built from. A search the answer did not rely on adds no citation.
    """
    by_key = {(c["doc_id"], c["section"]): c for c in candidates}
    answer_terms = significant_terms(answer)
    text_of = {(x["doc_id"], x["section"]): x["text"] for x in excerpts}
    chosen = []
    for key in named or []:
        # A section the model lists must also show up in what it wrote; this drops over-listed sources.
        if (key in by_key and by_key[key] not in chosen
                and len(answer_terms & significant_terms(text_of.get(key, ""))) >= MIN_NAMED_OVERLAP):
            chosen.append(by_key[key])
    best, best_score = None, 0
    for excerpt in excerpts:
        score = len(answer_terms & significant_terms(excerpt["text"]))
        if score > best_score and (excerpt["doc_id"], excerpt["section"]) in by_key:
            best, best_score = by_key[(excerpt["doc_id"], excerpt["section"])], score
    if best and best_score >= MIN_ANSWER_OVERLAP and best not in chosen:
        chosen.insert(0, best)
    return chosen


def _citation_merge(current, result):
    seen = {(x.get("doc_id"), x.get("section")) for x in current}
    for citation in result.get("citations", []):
        key = (citation.get("doc_id"), citation.get("section"))
        if key not in seen:
            current.append(citation); seen.add(key)


def _message_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [block["text"] for block in content
                 if isinstance(block, dict) and isinstance(block.get("text"), str)]
        return "\n".join(parts).strip()
    return str(content).strip()


_INLINE_POLICY_CITATION = re.compile(r"\s*\([^)]*(?:_policy|policy|faq)[^)]*§[^)]*\)", re.I)


def _remove_inline_policy_citations(answer: str) -> str:
    """Keep policy references in the API citations field, not in the prose."""
    def drop(match):
        # Removing "(policy §x)" must not glue its neighbours together ("from1 August", "covered.I can't").
        before, after = answer[match.start() - 1:match.start()], answer[match.end():match.end() + 1]
        return " " if before and after and not before.isspace() and not after.isspace() and after not in ".,;:!?)" else ""
    return _INLINE_POLICY_CITATION.sub(drop, answer).rstrip()


_RETURN_STOPWORDS = {"a","an","and","can","do","i","is","it","me","my","of","please","product","products","return","returning","the","this","that","to","want","which","with","item","items","back","send"}
# Word stems, so "defectiveness", "defect", "damage" and "broke" count as well as the exact words.
_RETURN_REASON_CHANGED = re.compile(r"\b(?:ch[ae]ng\w*\s+(?:of\s+|my\s+|the\s+)?mind|no\s+longer\s+(?:want|need)\w*|(?:don'?t|do\s+not)\s+(?:want|need|like)\w*)\b", re.I)
_RETURN_REASON_DEFECTIVE = re.compile(r"\b(?:defect\w*|fault\w*|broke\w*|broken|damag\w*|wrong\s+item|not\s+working|stopped\s+working|doesn'?t\s+work)\b", re.I)
_RETURN_DETAIL_CHANGE = re.compile(r"\b(?:actually|instead|switch|change|replace|use)\b", re.I)
_REASON_ASK = "the reason for the return: did they change their mind, or is the item defective"
_STOP_ASK = "whether they want to stop the return request for this item (a yes/no question: yes stops it, no keeps it open)"
_STOP_REQUEST = re.compile(r"\b(?:cancel|stop|never\s?mind|forget\s+it)\b", re.I)


def _order_not_found_turn(customer_id: str, message: str, entries: list[dict], prior_records: list[dict]):
    """A referenced order that is not on this account gets the same ORDER_NOT_FOUND reply as an unknown one."""
    reference = re.search(r"\bSS-\d+\b", message, re.I)
    if not reference or any(entry["order_id"].casefold() == reference.group(0).casefold() for entry in entries):
        return None
    order_id = reference.group(0).upper()
    result = build_tools(customer_id, lambda _: None, allow_proposals=False)[0].invoke({"order_id": order_id})
    if result.get("error_code") != "ORDER_NOT_FOUND":
        return None
    record = {"name": "get_orders", "args": {"order_id": order_id}, "status": "error"}
    return _ORDER_NOT_FOUND_ANSWER, "no_answer", prior_records + [record], [], False, None


# A return the rules refuse is a valid answer from a tool that ran fine; only a missing order, bad arguments or an
# internal failure is an error.
_RETURN_REJECTIONS = {"RETURN_WINDOW_EXPIRED", "NON_RETURNABLE", "NOT_DELIVERED", "OPENED_NOT_RETURNABLE"}


def _check_status(result: dict) -> str:
    """tool_calls status of a check_return_eligibility result."""
    if result.get("error") or (result.get("error_code") and result["error_code"] not in _RETURN_REJECTIONS):
        return "error"
    return "success"


def _return_reason(text: str) -> str | None:
    if _RETURN_REASON_CHANGED.search(text):
        return "changed_mind"
    if _RETURN_REASON_DEFECTIVE.search(text):
        return "defective"
    return None


def _last_assistant_text(history: list[Any]) -> str:
    for message in reversed(history):
        if isinstance(message, AIMessage):
            return _message_text(message.content)
    return ""


def _last_context(history: list[Any]) -> dict:
    """What the previous reply was waiting for, stored with that turn (see replies.Reply.context)."""
    for message in reversed(history):
        if isinstance(message, AIMessage):
            return (message.additional_kwargs or {}).get("context") or {}
    return {}


def _return_context(selected=None, explicit=False, awaiting=None, reason=None) -> dict:
    return {"flow": "return", "selected": selected, "explicit": explicit, "awaiting": awaiting, "reason": reason}


def _facts(situation: str, context: dict | None = None, must: list[str] | None = None, **data) -> dict:
    """What happened this turn, for the model to put into words. No customer-facing sentences live in Python."""
    facts = {"situation": situation, **data}
    if context is not None:
        facts["_context"] = context
    if must:
        facts["_must"] = must
    return facts


def _reason_in_previous_message(history: list[Any]) -> str | None:
    """For a reply to wording the model wrote itself (no stored context): the reason in the customer's last message only."""
    for message in reversed(history):
        if isinstance(message, HumanMessage):
            return _return_reason(_message_text(message.content))
    return None


def _item_ref(entry: dict) -> dict:
    return {"name": entry["name"], "order_id": entry["order_id"], "item_id": entry["item_id"]}


def _order_entries(result: dict) -> list[dict]:
    entries = []
    for order in result.get("orders", []):
        for item in order.get("items", []):
            entries.append({"order_id":order["order_id"],"status":order["status"],
                            "item_id":item["item_id"],"name":item["name"]})
    return entries


def _account_items(customer_id: str):
    tool = build_tools(customer_id, lambda _: None, allow_proposals=False)[0]
    return tool, _order_entries(tool.invoke({}))


def _lookup_account(customer_id: str, order_id: str | None = None):
    """Run get_orders (filtered when the customer named an order); return its record and item entries."""
    tool = build_tools(customer_id, lambda _: None, allow_proposals=False)[0]
    args = {"order_id": order_id} if order_id else {}
    result = tool.invoke(args)
    record = {"name": "get_orders", "args": args, "status": "error" if result.get("error_code") else "success"}
    return record, _order_entries(result)


def _return_window_policy(customer_id: str):
    """Run search_policy for the return window; cite only its best-matching clause, which the proposal relies on."""
    args = {"query": "return window"}
    result = build_tools(customer_id, lambda _: None, allow_proposals=False)[1].invoke(args)
    record = {"name": "search_policy", "args": args, "status": "error" if result.get("error") else "success"}
    return record, (result.get("citations", [])[:1] if result.get("covered") else [])


def _delivery_data(order_id: str) -> dict:
    """Days since delivery and the return window, both read at runtime (shipments and policy_parameters)."""
    with connect() as db:
        row = db.execute("SELECT delivered_at FROM shipments WHERE order_id=?", (order_id,)).fetchone()
        window = policy_parameter(db, "return_window_days")
    if not row or not row["delivered_at"] or window is None:
        return {}
    return {"days_since_delivery": (TODAY - date.fromisoformat(row["delivered_at"][:10])).days, "return_window_days": int(window)}


def _return_answer_without_reason(customer_id: str, item: dict, explicit: bool):
    """Facts on whether the item can be returned, without guessing the reason.

    The window, delivery and returnability checks do not depend on the reason, so a "no" there is final.
    Both reasons are checked by Python; neither is assumed for the customer.
    """
    order_id = item["order_id"]
    policy_record, policy_citations = _return_window_policy(customer_id)
    check = build_tools(customer_id, lambda _: None, allow_proposals=False)[2]
    records, outcome = [policy_record], {}
    for reason in ("changed_mind", "defective"):
        args = {"order_id": order_id, "item_ids": [item["item_id"]], "reason": reason}
        outcome[reason] = check.invoke(args)
        records.append({"name": "check_return_eligibility", "args": args,
                        "status": _check_status(outcome[reason])})
    changed, defective = outcome["changed_mind"], outcome["defective"]
    delivery = _delivery_data(order_id)
    if not defective.get("eligible"):
        expired = defective.get("error_code") == "RETURN_WINDOW_EXPIRED"
        facts = _facts("The customer asked whether the item can be returned. It cannot be returned, whatever the reason.", {},
                       item=item, can_be_returned=False, error_code=defective.get("error_code"), detail=defective.get("message"), **delivery)
        return facts, "answer", records, (policy_citations if expired else [])
    data = {"can_be_returned": True, "can_be_returned_if_defective": True, "can_be_returned_for_change_of_mind": bool(changed.get("eligible")), **delivery}
    if not changed.get("eligible"):
        data["change_of_mind_problem"] = changed.get("message")
    facts = _facts("The customer asked whether the item can be returned but has not said why. The checks above are final.",
                   _return_context(item, explicit, "reason"), item=item, ask=[_REASON_ASK], **data)
    return facts, "clarify", records, policy_citations


def _revise_pending_return(customer_id: str, proposal: dict, message: str):
    """Revalidate changed return details and make a new proposal, never a write."""
    reason=_return_reason(message)
    change_signal=bool(_RETURN_DETAIL_CHANGE.search(message))
    item_identifier=bool(re.search(r"\bSS-\d+(?:-\d+)?\b",message,re.I))
    if not reason and not change_signal:
        return None

    order_id=proposal["order_id"]
    item_ids=list(proposal["item_ids"])
    records=[]
    if item_identifier or (change_signal and reason is None):
        get_orders,entries=_account_items(customer_id)
        records.append(_get_orders_record(get_orders))
        not_found=_order_not_found_turn(customer_id,message,entries,records)
        if not_found is not None:
            return not_found
        matches=_matching_account_items(message,entries)
        unique={(entry["order_id"],entry["item_id"]):entry for entry in matches}
        matches=list(unique.values())
        if len(matches)!=1:
            facts=_facts("The customer wants to change which item is returned, but the item is unclear.",
                         _return_context(None,True,"item",proposal["reason"]),
                         matching_items=[_item_ref(entry) for entry in matches],
                         account_items=[] if matches else [_item_ref(entry) for entry in entries],
                         ask=["which item they mean"])
            return facts,"clarify",records,[],False,None
        order_id=matches[0]["order_id"]
        item_ids=[matches[0]["item_id"]]

    reason=reason or proposal["reason"]
    args={"order_id":order_id,"item_ids":item_ids,"reason":reason}
    tools=build_tools(customer_id,lambda _:None,allow_proposals=True)
    checked=tools[2].invoke(args)
    records.append({"name":"check_return_eligibility","args":args,
                    "status":_check_status(checked)})
    if not checked.get("eligible"):
        facts=_facts("The changed return is not possible. Nothing is waiting for confirmation and nothing was submitted.",{},
                     order_id=order_id,item_ids=item_ids,reason_given=reason,
                     error_code=checked.get("error_code"),detail=checked.get("message"))
        return facts,"answer",records,[],False,None
    proposal_holder={}
    tools=build_tools(customer_id,lambda value:proposal_holder.update(value=value),allow_proposals=True)
    created=tools[3].invoke(args)
    records.append({"name":"create_return_request","args":args,
                    "status":"pending_confirmation" if created.get("pending_confirmation") else "error"})
    if not created.get("pending_confirmation"):
        facts=_facts("The changed return could not be prepared. Nothing was submitted.",{},
                     error_code=created.get("error_code"),detail=created.get("message"))
        return facts,"answer",records,[],False,None
    facts=_facts("The return proposal was updated and waits for the customer's confirmation. Nothing is saved yet.",{},
                 must=[_CONFIRM_QUESTION],order_id=order_id,item_ids=item_ids,reason=reason,end_with=_CONFIRM_QUESTION)
    return facts,"answer",records,[],True,proposal_holder.get("value")


def _item_terms(value: str) -> set[str]:
    terms=set()
    for word in re.findall(r"[a-z0-9]+",value.casefold()):
        if len(word)>2 and word not in _RETURN_STOPWORDS:
            terms.add(word[:-1] if word.endswith("s") and len(word)>4 else word)
    return terms


def _matching_account_items(message: str, entries: list[dict]) -> list[dict]:
    item_id = re.search(r"\bSS-\d+-\d+\b",message,re.I)
    if item_id:
        return [entry for entry in entries if entry["item_id"].casefold()==item_id.group(0).casefold()]
    order_id = re.search(r"\bSS-\d+\b",message,re.I)
    if order_id:
        in_order=[entry for entry in entries if entry["order_id"].casefold()==order_id.group(0).casefold()]
        # An item described next to its order ID narrows a multi-item order.
        described=_item_terms(re.sub(r"\bSS-\d+(?:-\d+)?\b"," ",message))
        scored=[(len(described & _item_terms(entry["name"])),entry) for entry in in_order]
        best=max((score for score,_entry in scored),default=0)
        return [entry for score,entry in scored if score==best] if best else in_order
    terms=_item_terms(message)
    if not terms:
        return []
    scored=[(len(terms & _item_terms(entry["name"])),entry) for entry in entries]
    best=max((score for score,_entry in scored),default=0)
    return [entry for score,entry in scored if score==best and score>0]


def _get_orders_record(tool):
    return {"name":"get_orders","args":{},"status":"success"}


def _return_detail_turn(customer_id: str, history: list[Any], message: str, pending: dict | None, context_override: dict | None = None):
    """Resolve customer-described items from their account without guessing an order ID or reason.

    Returns facts (not prose) for the reply, plus the route, tool-call records, citations and any pending proposal.
    """
    if pending:
        return None
    last_answer=_last_assistant_text(history)
    context=context_override if context_override is not None else _last_context(history)
    in_flow=context.get("flow")=="return"
    selected=context.get("selected") if in_flow else None
    reason=_return_reason(message) or (context.get("reason") if in_flow else _reason_in_previous_message(history))
    awaiting=context.get("awaiting") if in_flow else None
    short_reply=len(re.findall(r"[a-z0-9']+",message,re.I))<=6 and "?" not in message
    explicit_flow=bool(in_flow and context.get("explicit"))

    # A bare "no" does not answer "changed your mind or defective?": it could mean "stop" or "neither of those".
    # Verify before dropping the return, and keep the selected item so the flow can carry on after the answer.
    if awaiting=="reason" and selected and not _return_reason(message):
        if short_reply and _STOP_REQUEST.search(message):
            return _facts("The customer asked to stop the return. Nothing was submitted.",{}),"answer",[],[],False,None
        if is_clear_decline(message):
            facts=_facts("The customer answered 'no' to the question about the reason, so it is unclear whether they want to stop the return. "
                         "Nothing was submitted and nothing is dropped yet. Do not assume either.",
                         _return_context(selected,explicit_flow,"stop"),item=selected,ask=[_STOP_ASK])
            return facts,"clarify",[],[],False,None
        # A short reply without a reason ("yes") is not a new request.
        if short_reply:
            facts=_facts("The customer replied, but not with a return reason, so the return cannot be checked yet. Do not assume a reason.",
                         _return_context(selected,explicit_flow,"reason"),item=selected,ask=[_REASON_ASK])
            return facts,"clarify",[],[],False,None

    # The previous reply asked whether to stop: yes stops, no carries on (the reason is still needed), a reason resumes.
    if awaiting=="stop" and selected and not _return_reason(message):
        if is_clear_confirmation(message) or (short_reply and _STOP_REQUEST.search(message)):
            return _facts("The customer confirmed they want to stop the return. Nothing was submitted.",{}),"answer",[],[],False,None
        if short_reply:
            leading_no=bool(re.match(r"\s*(?:no|nope|nah)\b",message,re.I))
            if is_clear_decline(message) or leading_no or allows_new_proposal(message,None):
                facts=_facts("The customer does not want to stop, so the return carries on, but the reason is still needed. Do not assume one.",
                             _return_context(selected,explicit_flow,"reason"),item=selected,ask=[_REASON_ASK])
            else:
                facts=_facts("The customer's reply did not say whether to stop the return. Nothing was submitted.",
                             _return_context(selected,explicit_flow,"stop"),item=selected,ask=[_STOP_ASK])
            return facts,"clarify",[],[],False,None

    if is_clear_decline(message) and _RETURN_OFFER.search(last_answer):
        return _facts("The customer declined the return you offered. Nothing was submitted.",{}),"answer",[],[],False,None

    accepted_offer=is_clear_confirmation(message) and bool(_RETURN_OFFER.search(last_answer))
    explicit_action=allows_new_proposal(message,None) or bool(in_flow and context.get("explicit")) or accepted_offer

    # A yes to an assistant's general offer expresses interest, not final
    # confirmation. Resolve the offered item against this customer's account
    # and carry that intent forward while asking for any missing reason.
    offer_lookup_calls=[]
    if accepted_offer:
        get_orders,entries=_account_items(customer_id)
        offer_lookup_calls=[_get_orders_record(get_orders)]
        if selected:
            matches=[entry for entry in entries if entry["order_id"]==selected["order_id"] and entry["item_id"]==selected["item_id"]]
        else:
            matches=_matching_account_items(last_answer,entries)
            if not matches:
                for prior in reversed(history):
                    if isinstance(prior,HumanMessage):
                        prior_text=_message_text(prior.content)
                        if re.search(r"\b(?:return|send.{0,12}back)\b",prior_text,re.I):
                            matches=_matching_account_items(prior_text,entries)
                            if matches:
                                break
        unique_matches={(entry["order_id"],entry["item_id"]):entry for entry in matches}
        matches=list(unique_matches.values())
        if len(matches)==1:
            selected=_item_ref(matches[0])
        else:
            facts=_facts("The customer said yes to returning something, but the item is unclear.",
                         _return_context(None,True,"item",reason),
                         matching_items=[_item_ref(entry) for entry in matches],
                         account_items=[] if matches else [_item_ref(entry) for entry in entries],
                         ask=["which item they mean"]+([] if reason else [_REASON_ASK]))
            return facts,"clarify",offer_lookup_calls,[],False,None

    # Context carried from earlier turns applies only to a message that is itself about the return;
    # an unrelated question must not revive an old selection or offer.
    answering_item=awaiting=="item" and (short_reply or bool(re.search(r"\bSS-\d+",message,re.I)))
    engaged=(context_override is not None or bool(_return_reason(message)) or accepted_offer
             or bool(re.search(r"(?:return|send.{0,12}back)",message,re.I)) or answering_item)

    # A selected item and a stated reason are sufficient for Python eligibility.
    if selected and reason and engaged:
        tools=build_tools(customer_id,lambda _:None,allow_proposals=explicit_action)
        check_tool,create_tool=tools[2],tools[3]
        args={"order_id":selected["order_id"],"item_ids":[selected["item_id"]],"reason":reason}
        policy_record,policy_citations=_return_window_policy(customer_id)
        checked=check_tool.invoke(args)
        records=[policy_record,{"name":"check_return_eligibility","args":args,
                                "status":_check_status(checked)}]
        delivery=_delivery_data(selected["order_id"])
        if not checked.get("eligible"):
            # The window clause is cited only when the window is the reason for the refusal.
            cited=policy_citations if checked.get("error_code")=="RETURN_WINDOW_EXPIRED" else []
            facts=_facts("The item cannot be returned. Nothing is waiting for confirmation and nothing was submitted.",{},
                         item=selected,reason_given=reason,can_be_returned=False,error_code=checked.get("error_code"),detail=checked.get("message"),**delivery)
            return facts,"answer",offer_lookup_calls+records,cited,False,None
        if explicit_action:
            proposal_holder={}
            tools=build_tools(customer_id,lambda proposal:proposal_holder.update(value=proposal),allow_proposals=True)
            result=tools[3].invoke(args)
            records.append({"name":"create_return_request","args":args,
                            "status":"pending_confirmation" if result.get("pending_confirmation") else "error"})
            if not result.get("pending_confirmation"):
                facts=_facts("The return could not be prepared. Nothing was submitted.",{},
                             item=selected,error_code=result.get("error_code"),detail=result.get("message"))
                return facts,"answer",offer_lookup_calls+records,[],False,None
            facts=_facts("The item is eligible and a return request is prepared. It waits for the customer's confirmation; nothing is saved yet.",{},
                         must=[_CONFIRM_QUESTION],item=selected,reason=reason,eligible_for_return=True,end_with=_CONFIRM_QUESTION,**delivery)
            return facts,"answer",offer_lookup_calls+records,policy_citations,True,proposal_holder.get("value")
        # The customer only asked a question, so nothing is pending (pending_confirmation stays false). The specific
        # offer is remembered, so one clear "yes" to it goes through the normal revalidated write, with no second prompt.
        offer={"kind":"return","order_id":selected["order_id"],"item_ids":[selected["item_id"]],"reason":reason,"offered":True}
        facts=_facts("The customer asked whether they can return the item. It is eligible. No request has been made yet; if the customer says yes, you will submit it.",{},
                     item=selected,reason=reason,eligible_for_return=True,ask=["whether they want you to submit the return request now"],**delivery)
        return facts,"answer",offer_lookup_calls+records,policy_citations,False,offer

    # A previous clarification carries the selected database item through the
    # session context. If the reason is still missing, say what is known and ask for it.
    if selected and not reason and engaged:
        facts,route,records,cited=_return_answer_without_reason(customer_id,selected,explicit_action)
        return facts,route,offer_lookup_calls+records,cited,False,None

    broad_reference=bool(_RETURN_BROAD_REFERENCE.search(message))
    specific_reference=bool(re.search(r"\b(?:return|send)\s+(?:my|the|this|that)\s+[a-z0-9][a-z0-9 -]{0,50}",message,re.I)
                            or re.search(r"\b(?:can|could|may)\s+i\s+(?:return|send)\s+(?:my|the|this|that)\s+[a-z0-9]",message,re.I))
    return_operation=bool(re.search(r"\b(?:return|send.{0,12}back)\b",message,re.I))
    general_return_reference=(return_operation and not re.search(r"\b(?:policy|window|period|days|how long|request status|tracking|when|status)\b",message,re.I))
    # Only a reply that actually asked the customer something can be answered; a policy statement cannot.
    asked_customer=last_answer.rstrip().endswith("?") or bool(_REQUESTS_DETAILS.search(last_answer))
    # An answer is short or names an order; a full new question (for example about warranty) is not one.
    answering_return_clarification=answering_item or bool(asked_customer and _RETURN_CLARIFICATION_CONTEXT.search(last_answer)
                                                          and (short_reply or re.search(r"\bSS-\d+",message,re.I)))
    is_return_request=broad_reference or specific_reference or general_return_reference or answering_return_clarification
    if not (awaiting=="item" or is_return_request):
        return None

    # When the customer names an order, look up that order: an unknown or foreign one fails here, as ORDER_NOT_FOUND.
    order_ref=re.search(r"\bSS-\d+\b",message,re.I)
    get_orders_call,entries=_lookup_account(customer_id,order_ref.group(0).upper() if order_ref else None)
    if get_orders_call["status"]=="error":
        return _ORDER_NOT_FOUND_ANSWER,"no_answer",[get_orders_call],[],False,None
    if broad_reference:
        facts=_facts("The customer wants to return something but has not said which item.",
                     _return_context(None,explicit_action,"item",reason),
                     account_items=[_item_ref(entry) for entry in entries],
                     ask=["which item they mean"]+([] if reason else [_REASON_ASK]))
        return facts,"clarify",[get_orders_call],[],False,None

    matches=_matching_account_items(message,entries)
    unique={(entry["order_id"],entry["item_id"]):entry for entry in matches}
    matches=list(unique.values())
    if len(matches)!=1:
        facts=_facts("The customer wants to return an item, but it is unclear which one." if matches
                     else "The customer wants to return an item, but nothing on their account matches that description.",
                     _return_context(None,explicit_action,"item",reason),
                     matching_items=[_item_ref(entry) for entry in matches],
                     account_items=[] if matches else [_item_ref(entry) for entry in entries],
                     ask=["which item they mean"]+([] if reason else [_REASON_ASK]))
        return facts,"clarify",[get_orders_call],[],False,None

    item=_item_ref(matches[0])
    if not reason:
        facts,route,records,cited=_return_answer_without_reason(customer_id,item,explicit_action)
        return facts,route,[get_orders_call]+records,cited,False,None
    # Re-enter the shared Python eligibility path with an explicit selected item.
    result=_return_detail_turn(customer_id,history,message,None,
                               context_override=_return_context(item,explicit_action,None,reason))
    if result is None:
        return None
    answer,route,calls,citations,pending_value,proposal=result
    return answer,route,[get_orders_call]+calls,citations,pending_value,proposal


def _build_graph(customer_id: str, model, proposal_holder: dict, allow_proposals: bool):
    tools = build_tools(customer_id, lambda p: proposal_holder.update(value=p), allow_proposals)
    tool_by_name = {t.name: t for t in tools}

    def agent_node(state: AgentState):
        if state.get("tool_limit_reached"):
            ai = model.invoke(state["messages"] + [SystemMessage(content=(
                "The per-message tool limit was reached. Call no more tools. Tell the customer you could not check "
                "everything in one request, and ask which item they mean and, for a return, whether they changed "
                "their mind or the item is defective."))])
            return {"messages": [ai], "call_count": state.get("call_count", 0)}
        ai = model.bind_tools(tools if allow_proposals else tools[:3]).invoke(state["messages"])
        return {"messages": [ai], "call_count": state.get("call_count", 0)}

    def route(state: AgentState):
        latest = state["messages"][-1]
        # Even after the cumulative execution budget is exhausted, route an
        # emitted batch through tools_node so every call ID gets a result.
        # tools_node marks over-budget calls as skipped; agent_node then
        # returns a deterministic clarification without another LLM request.
        return "tools" if getattr(latest, "tool_calls", None) else END

    def tools_node(state: AgentState):
        response_messages, records = [], list(state.get("tool_calls", []))
        citations = list(state.get("citations", []))
        excerpts = list(state.get("excerpts", []))
        order_not_found = state.get("order_not_found", False)
        policy_searched = state.get("policy_searched", False)
        policy_covered = state.get("policy_covered", False)
        order_found = state.get("order_found", False)
        warranty_orders = list(state.get("warranty_orders", []))
        non_policy_tool_used = state.get("non_policy_tool_used", False)
        pending_calls = state["messages"][-1].tool_calls
        budget = max(0, MAX_TOOL_CALLS - state.get("call_count", 0))
        executed_count = min(len(pending_calls), budget)
        tool_limit_reached = len(pending_calls) > budget
        for index, call in enumerate(pending_calls):
            name, args = call["name"], call["args"]
            if index >= budget:
                result = {"error_code": "TOOL_CALL_LIMIT", "message": "This tool call was not run because the per-message tool limit was reached."}
                status = "error"
            else:
                try:
                    if name not in tool_by_name:
                        raise ValueError("Unknown tool")
                    result = tool_by_name[name].invoke(args)
                    if name == "search_policy":
                        _citation_merge(citations, result)
                        excerpts.extend(result.get("excerpts", []))
                        policy_searched = True
                        policy_covered = policy_covered or bool(result.get("covered"))
                    else:
                        non_policy_tool_used = True
                    if name == "get_orders":  # this customer's delivered orders with an item under warranty, for a ticket offer
                        warranty_orders += [o["order_id"] for o in result.get("orders", [])
                                            if o["status"] == "delivered" and any(i["warranty_active"] for i in o["items"])]
                    if result.get("error_code") == "ORDER_NOT_FOUND":
                        order_not_found = True
                    elif name != "search_policy" and not result.get("error_code") and not result.get("error"):
                        order_found = True
                    status = "pending_confirmation" if result.get("pending_confirmation") else (_check_status(result) if name == "check_return_eligibility" else ("error" if result.get("error_code") or result.get("error") else "success"))
                except Exception as exc:
                    result, status = {"error": "TOOL_ERROR", "message": str(exc)}, "error"
            records.append({"name": name, "args": args, "status": status})
            response_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {"messages": response_messages, "tool_calls": records, "citations": citations, "excerpts": excerpts,
                "proposal": proposal_holder.get("value"), "order_not_found": order_not_found,
                "policy_searched": policy_searched, "policy_covered": policy_covered,
                "policy_uncovered": policy_searched and not policy_covered,
                "order_found": order_found, "non_policy_tool_used": non_policy_tool_used, "warranty_orders": warranty_orders,
                "call_count": state.get("call_count", 0) + executed_count,
                "tool_limit_reached": tool_limit_reached}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(), tools


def _fallback(message: str, customer_id: str):
    """Limited behavior with no API key: Python looks things up and hands back facts; never guesses account data."""
    from app.tools.support_tools import build_tools
    records = []
    tools = build_tools(customer_id, lambda _: None)
    order_match = re.search(r"\bSS-\d+\b", message, re.I)
    if order_match:
        order_id = order_match.group(0).upper()
        tool = next(t for t in tools if t.name == "get_orders")
        result = tool.invoke({"order_id": order_id})
        records.append({"name": "get_orders", "args": {"order_id": order_id}, "status": "error" if result.get("error_code") else "success"})
        if result.get("error_code"):
            return _ORDER_NOT_FOUND_ANSWER, "no_answer", records, [], False
        order = result["orders"][0]
        ship = order.get("shipment") or {}
        facts = _facts("The customer asked about this order.", order_id=order_id, status=order["status"],
                       estimated_delivery=ship.get("eta"), carrier=ship.get("carrier"), tracking_number=ship.get("tracking_no"))
        return facts, "answer", records, [], False
    if re.search(r"return|refund|warranty|shipping|cancel|exchange|policy|delivery", message, re.I):
        result = search_policy(message)
        records.append({"name": "search_policy", "args": {"query": message}, "status": "success" if result.get("covered") else "error"})
        if result.get("covered"):
            excerpt = result["excerpts"][0]
            return _facts("The customer asked a policy question. Answer from this policy excerpt only.", policy_excerpt=excerpt["text"]), "answer", records, result["citations"][:1], False
        return _facts("No current ShopSphere policy covers the question. Say so and suggest emailing the support team."), "no_answer", records, [], False
    if re.search(r"\b(hi|hello|hey|thanks|thank you)\b", message, re.I):
        return _facts("The customer greeted you. Say what you can help with: ShopSphere orders, returns and policies."), "answer", records, [], False
    return _facts("The message is not about ShopSphere. Say politely that you can only help with ShopSphere orders and policies."), "out_of_scope", records, [], False


def _order_in_recent_history(history: list[Any]) -> bool:
    """True when one of the last few messages names an order, so a bare 'when will it arrive?' has something to refer to."""
    return any(re.search(r"\bSS-\d+\b", normalize_order_ids(_message_text(m.content)), re.I) for m in history[-6:])


_TICKET_OFFER = re.compile(r"\b(?:would you like|shall i|should i|do you want|want me to)\b[^?.]{0,80}\b(?:open|file|raise|create|submit|start|set up)\b"
                          r"[^?.]{0,40}\bwarranty\b[^?.]{0,20}\b(?:ticket|claim|request)\b", re.I)


def _warranty_ticket_offer(customer_id: str, answer: str, message: str, orders: list[str]):
    """Remember a warranty ticket the assistant just offered, so the customer's "yes" has something to confirm.

    Python chooses the order (one of this customer's own, under warranty) and the summary (the customer's own words);
    nothing is written here, and the write is revalidated when the customer confirms.
    """
    if not _TICKET_OFFER.search(answer):
        return None
    named = [order_id for order_id in dict.fromkeys(orders) if order_id.casefold() in answer.casefold()]
    candidates = named or list(dict.fromkeys(orders))
    if len(candidates) != 1:
        return None
    summary = " ".join(message.split())[:300]
    if not summary or not validate_ticket(customer_id, candidates[0], "warranty", summary)["eligible"]:
        return None
    return {"kind": "ticket", "order_id": candidates[0], "ticket_type": "warranty", "summary": summary, "offered": True}


def _llm_answer(customer_id: str, history: list[Any], message: str, previous_proposal: dict | None):
    settings = get_settings()
    extra = {"reasoning_effort": settings.openai_reasoning_effort} if settings.openai_reasoning_effort else {}
    proposal_holder: dict = {}
    model = ChatOpenAI(model=settings.openai_model, temperature=0, api_key=settings.openai_api_key,
                       timeout=10, max_retries=0, **extra)
    compiled, tools = _build_graph(customer_id, model, proposal_holder, allows_new_proposal(message, previous_proposal))
    # The business date, so the model can tell a policy already in force from a future one.
    business_today = date.fromisoformat(settings.app_today)
    today_sentence = f" Today's date is {business_today.day} {business_today:%B %Y}; a policy whose effective date is on or before today is in force."

    # The scripted return flow already took any message that answers its question, so a message that reaches the
    # model while a question is still open is a new one: the old question must not be repeated in its place.
    stale = ()
    if _last_context(history).get("awaiting"):
        stale = (SystemMessage(content="Your previous question to the customer is closed. Answer only the customer's latest message, "
                                       "calling the tools it needs, and do not repeat or continue the earlier question."),)

    def run(*extra_system):
        messages = [SystemMessage(content=SYSTEM + today_sentence), *extra_system] + history[-12:] + [*stale, HumanMessage(content=message)]
        return compiled.invoke({"messages": messages, "tool_calls": [], "citations": [], "call_count": 0},
                               config={"recursion_limit": 18})

    # Ownership of every order the customer named is decided in Python, whatever the model looks up (or invents).
    named_ids = {m.upper() for m in re.findall(r"\bSS-\d+\b", message, re.I)}
    with connect() as db:
        missing = {order_id for order_id in named_ids if not owned_order(db, order_id, customer_id)}

    try:
        result = run()
        _text, first_route = _split_route(_message_text(result["messages"][-1].content))
        # The model called the question off-topic without looking anything up, yet the policy files do have matching
        # text: the question is in scope. Let it try once more, told to search.
        if first_route == "out_of_scope" and not result.get("tool_calls") and search_policy(message).get("covered"):
            result = run(SystemMessage(content="This message matches ShopSphere policy text, so it is in scope. Call search_policy and answer from its excerpts."))
        answer, route_hint = _split_route(_message_text(result["messages"][-1].content))
        answer, named_sources = _split_sources(answer)
        answer = _remove_inline_policy_citations(answer)
        records = result.get("tool_calls", [])
        proposal = proposal_holder.get("value")
        if missing:
            other_part_answered = (result.get("order_found") and len(named_ids) > len(missing)) or result.get("policy_covered")
            if not other_part_answered:
                return _ORDER_NOT_FOUND_ANSWER, "no_answer", records, [], False, None
            # Another part of the message was answerable: keep it and say plainly that the other order was not found.
            if _ORDER_NOT_FOUND_ANSWER.casefold() not in answer.casefold():
                answer = f"{answer} {_ORDER_NOT_FOUND_ANSWER}".strip()
        if route_hint == "out_of_scope" and not result.get("order_found"):
            return answer, "out_of_scope", [], [], False, None
        if result.get("policy_uncovered"):
            return answer, "no_answer", records, [], bool(proposal), proposal
        if proposal and _CONFIRM_QUESTION.casefold() not in answer.casefold():
            answer = f"{answer} {_CONFIRM_QUESTION}".strip()
        citations = _choose_citations(named_sources, result.get("citations", []), result.get("excerpts", []), answer)
        if not answer:
            return _UNAVAILABLE_REPLY, "no_answer", records, [], False, None
        offer = None if proposal else _warranty_ticket_offer(customer_id, answer, message, result.get("warranty_orders", []))
        return answer, ("clarify" if route_hint == "clarify" else "answer"), records, citations, bool(proposal), proposal or offer
    except Exception as exc:
        if adapt_to_model_error(exc, settings):  # the model needed reasoning switched off: try again once with that
            return _llm_answer(customer_id, history, message, previous_proposal)
        if named_ids and missing == named_ids:  # nothing the customer named is theirs: the answer needs no model
            return _ORDER_NOT_FOUND_ANSWER, "no_answer", [], [], False, None
        return _UNAVAILABLE_REPLY, "no_answer", [], [], False, None


def run_chat(customer_id: str, history: list[Any], message: str, pending: dict | None):
    """One chat turn: Python decides, and the model puts the result into words (see replies.py)."""
    message = normalize_order_ids(add_default_year(message))
    result = _run_turn(customer_id, history, message, pending)
    if isinstance(result[0], dict):
        result = (compose_reply(history, message, result[0], ChatOpenAI), *result[1:])
    return result


_ORDER_DATE_INTENT = re.compile(r"\b(?:i\s+)?(?:ordered|bought|purchased)\b|\border(?:s)?\s+(?:placed\s+)?(?:on|dated|from)\b", re.I)


def _order_date_turn(customer_id: str, message: str):
    """Which of this customer's orders was placed on the date they gave. Only their own orders are searched."""
    order_date = find_date(message)
    if not order_date or re.search(r"\bSS-\d+\b", message, re.I) or not _ORDER_DATE_INTENT.search(message):
        return None
    args = {"order_date": order_date}
    orders = build_tools(customer_id, lambda _: None, allow_proposals=False)[0].invoke(args).get("orders", [])
    record = {"name": "get_orders", "args": args, "status": "success" if orders else "error"}
    if not orders:
        return _ORDER_NOT_FOUND_ANSWER, "no_answer", [record], [], False, None
    choices = [{"order_id": o["order_id"], "status": o["status"], "products": [i["name"] for i in o["items"]]} for o in orders]
    if len(choices) > 1:
        facts = _facts("The customer asked which order they placed on a date, and several of their orders match. Do not guess.",
                       order_date=order_date, matching_orders=choices, ask=["which order they mean"])
        return facts, "clarify", [record], [], False, None
    facts = _facts("The customer asked what they ordered on a date; this is their only order from that day.", order_date=order_date, order=choices[0])
    return facts, "answer", [record], [], False, None


def _run_turn(customer_id: str, history: list[Any], message: str, pending: dict | None):
    previous_proposal = pending
    compound = _is_compound(message)
    if pending:
        if is_clear_confirmation(message):
            result = execute_proposal(customer_id, pending)
            name = "create_return_request" if pending["kind"] == "return" else "create_support_ticket"
            args = {k: v for k, v in pending.items() if k not in ("kind", "offered")}
            ok = bool(result.get("created"))
            record = {"name": name, "args": args, "status": "success" if ok else "error"}
            label = "return request" if pending["kind"] == "return" else "support ticket"
            if ok:
                id_value = result.get("request_id") or result.get("ticket_id")
                facts = _facts(f"The customer confirmed and the {label} was saved.", {}, must=[str(id_value)],
                               action=label, reference_id=id_value, **{k: v for k, v in args.items()})
                return facts, "answer", [record], [], False, None
            # Revalidation refused the write: a "no, because..." reply, not a missing-order or missing-policy one.
            if result.get("error_code") == "ORDER_NOT_FOUND":
                return _ORDER_NOT_FOUND_ANSWER, "no_answer", [record], [], False, None
            facts = _facts(f"The customer confirmed, but the {label} was refused when it was checked again, so nothing was saved.", {},
                           action=label, error_code=result.get("error_code"), detail=result.get("message"))
            return facts, "answer", [record], [], False, None
        if is_clear_decline(message):
            label = "return request" if pending["kind"] == "return" else "support ticket"
            return _facts(f"The customer declined. The {label} was not submitted.", {}), "answer", [], [], False, None
        if pending["kind"] == "return" and not compound and not pending.get("offered"):
            revised = _revise_pending_return(customer_id, pending, message)
            if revised is not None:
                return revised
        # Pending proposals are discarded by every non-affirmative next message.
        pending = None

    if re.fullmatch(r"(?:just\s+)?do it[.!\s]*", message, re.I):
        return (_facts("The customer said 'do it', but nothing is waiting for confirmation, so nothing was done.",
                       ask=["which ShopSphere action they want you to prepare"]), "clarify", [], [], False, None)

    by_date = None if compound else _order_date_turn(customer_id, message)
    if by_date is not None:
        return by_date

    # Resolve a missing order identifier before the LLM can choose an arbitrary
    # result from the optional unfiltered get_orders listing. The customer may
    # see their own matching order IDs so they can identify the intended one.
    # When the chat already named an order, the agent resolves "it" from history instead.
    if (not compound and not re.search(r"\bSS-\d+\b", message, re.I) and _ORDER_LOOKUP_INTENT.search(message)
            and not _order_in_recent_history(history)):
        get_orders = build_tools(customer_id, lambda _: None, allow_proposals=False)[0]
        result = get_orders.invoke({})
        ids = [order["order_id"] for order in result.get("orders", [])]
        facts = _facts("The customer asked about an order without saying which one.", orders_on_account=ids, ask=["which order"])
        call = {"name": "get_orders", "args": {}, "status": "success"}
        return facts, "clarify", [call], [], False, None

    # Scripted return shortcuts handle one return request; anything compound goes to the agent loop.
    return_detail = None if compound else _return_detail_turn(customer_id, history, message, pending)
    if return_detail is not None:
        return return_detail

    if not get_settings().openai_api_key:
        answer, route, records, citations, waiting = _fallback(message, customer_id)
        return answer, route, records, citations, waiting, None
    return _llm_answer(customer_id, history, message, previous_proposal)


