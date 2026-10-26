import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages

from app.core.config import get_settings
from app.core.dates import add_default_year
from app.services.policy_search import search_policy
from app.tools.support_tools import build_tools, execute_proposal

MAX_TOOL_CALLS = 6
_CONFIRMATIONS = {"yes", "yeah", "yep", "sure", "okay", "ok", "absolutely", "confirm", "confirmed", "go ahead", "please do", "yes go ahead", "yes please", "i confirm", "please proceed", "that s right"}
_ACTION_REQUEST = re.compile(r"\b(return(?:s|ing)?|cancel|cancellation|create|open|submit|file|request|ticket|claim)\b", re.I)
_DETAIL_CHANGE = re.compile(r"^(?:(?:actually[,]?\s+)?(?:just|only)\s+(?:the\s+)?|(?:actually[,]?\s+)?(?:change it to|switch (?:it )?to|replace (?:it )?(?:with|by)|make it)\s+(?:the\s+)?)[a-z0-9][a-z0-9 -]{1,50}$", re.I)
_ORDER_LOOKUP_INTENT = re.compile(r"\b(?:where(?:'s| is)|track(?:ing)?|status of|delivery (?:date|status)|eta|when (?:will|does) .{0,45} arrive|has .{0,45} shipped)\b", re.I)
_RETURN_BROAD_REFERENCE = re.compile(r"\b(?:my\s+(?:products?|items?)|all\s+(?:my\s+)?(?:products?|items?)|which\s+of\s+my\s+(?:products?|items?)|these\s+(?:products?|items?)|this\s+(?:item|product)|that\s+(?:item|product))\b", re.I)
_RETURN_OFFER = re.compile(r"\b(?:would you like me to|shall i|should i)\b.{0,100}\b(?:return|send.{0,12}back)\b", re.I | re.S)
_RETURN_ACTION_MARKER = re.compile(r"^You'd like to proceed with a return\b", re.I)
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
    if re.search(r"\?\s*$", text):
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
    proposal: dict | None
    customer_id: str
    tools: list[Any]
    call_count: int
    policy_uncovered: bool
    non_policy_tool_used: bool
    final_answer: str
    order_not_found: bool
    tool_limit_reached: bool


SYSTEM = """You are ResolveBot, ShopSphere customer support. Use tools for order facts and policy questions. Never invent facts, policies, IDs, or eligibility. Return eligibility is determined only by the Python tool. Never ask for or accept customer ID; it comes from trusted request context. Customers do not need to know an order ID: use get_orders to find their own products when appropriate. If an item or order is missing or several products/orders match, list the relevant account choices and ask which one; ask for every missing eligibility detail, including reason, together. Do not check every account item when a broad question has not identified an item. You may answer multiple questions in one message. For a direct policy answer, rely only on search_policy excerpts. Do not add document IDs, section numbers, or citation lists to the customer-facing answer; citations are returned separately in the API response. If coverage is missing, say ShopSphere policies do not cover that question and suggest emailing the support team. If an excerpt identifies a future policy, state its effective date and distinguish it from current policy. If a request is unrelated to ShopSphere, reply that you can only help with ShopSphere support. For a possible write, propose only when the customer clearly requests the action; questions like 'Can I return this?' ask for eligibility only. A proposal is not an execution: explain it and ask for explicit confirmation. Tool writes are gated by Python and confirmation. Customer dates without a year refer to 2026. Keep answers concise and customer-friendly."""


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
    return _INLINE_POLICY_CITATION.sub("", answer).rstrip()


_RETURN_STOPWORDS = {"a","an","and","can","do","i","is","it","me","my","of","please","product","products","return","returning","the","this","that","to","want","which","with","item","items","back","send"}
_RETURN_REASON_CHANGED = re.compile(r"\b(?:changed\s+my\s+mind|change\s+of\s+mind|no\s+longer\s+(?:want|need)|don't\s+want)\b", re.I)
_RETURN_REASON_DEFECTIVE = re.compile(r"\b(?:defective|broken|damaged|wrong\s+item|arrived\s+damaged)\b", re.I)
_RETURN_DETAIL_CHANGE = re.compile(r"\b(?:actually|instead|switch|change|replace|use)\b", re.I)
_SELECTED_RETURN_CONTEXT = re.compile(r"(?:I found your|^The|You'd like to proceed with a return for) (?P<name>.+?) \(order (?P<order>SS-\d+), item (?P<item>SS-\d+-\d+)\)", re.I)
_RETURN_REASON_QUESTION = "Was the reason that you changed your mind, or that the item is defective?"


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


def _selected_return_context(history: list[Any]) -> dict | None:
    match = _SELECTED_RETURN_CONTEXT.search(_last_assistant_text(history))
    if not match:
        return None
    return {"name":match.group("name"),"order_id":match.group("order"),"item_id":match.group("item")}


def _return_action_in_history(history: list[Any]) -> bool:
    if _RETURN_ACTION_MARKER.search(_last_assistant_text(history)):
        return True
    return any(
        isinstance(message, HumanMessage)
        and re.search(r"\breturn(?:s|ing)?\b", _message_text(message.content), re.I)
        and allows_new_proposal(_message_text(message.content), None)
        for message in history
    )


def _return_reason_in_history(history: list[Any]) -> str | None:
    for message in reversed(history):
        if isinstance(message, HumanMessage):
            reason = _return_reason(_message_text(message.content))
            if reason:
                return reason
    return None


def _account_items(customer_id: str):
    tool = build_tools(customer_id, lambda _: None, allow_proposals=False)[0]
    result = tool.invoke({})
    entries = []
    for order in result.get("orders", []):
        for item in order.get("items", []):
            entries.append({"order_id":order["order_id"],"status":order["status"],
                            "item_id":item["item_id"],"name":item["name"]})
    return tool, entries


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
        matches=_matching_account_items(message,entries)
        unique={(entry["order_id"],entry["item_id"]):entry for entry in matches}
        matches=list(unique.values())
        if len(matches)!=1:
            choices="; ".join(f"{entry['name']} (order {entry['order_id']}, item {entry['item_id']})" for entry in matches)
            answer=(f"You'd like to change the return item. Which one do you mean: {choices}?"
                    if choices else "I couldn't match that item to your account. Which product or order item do you mean?")
            return answer,"clarify",records,[],False,None
        order_id=matches[0]["order_id"]
        item_ids=[matches[0]["item_id"]]

    reason=reason or proposal["reason"]
    args={"order_id":order_id,"item_ids":item_ids,"reason":reason}
    tools=build_tools(customer_id,lambda _:None,allow_proposals=True)
    checked=tools[2].invoke(args)
    records.append({"name":"check_return_eligibility","args":args,
                    "status":"error" if checked.get("error_code") else "success"})
    if not checked.get("eligible"):
        return checked.get("message","That return is not eligible."),"answer",records,[],False,None
    proposal_holder={}
    tools=build_tools(customer_id,lambda value:proposal_holder.update(value=value),allow_proposals=True)
    created=tools[3].invoke(args)
    records.append({"name":"create_return_request","args":args,
                    "status":"pending_confirmation" if created.get("pending_confirmation") else "error"})
    if not created.get("pending_confirmation"):
        return created.get("message","I couldn't prepare that return."),"answer",records,[],False,None
    label=f"item {item_ids[0]}" if len(item_ids)==1 else "the selected items"
    answer=f"I updated the return proposal for {label} with reason {reason}. Please confirm again to submit it."
    return answer,"answer",records,[],True,proposal_holder.get("value")


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
        return [entry for entry in entries if entry["order_id"].casefold()==order_id.group(0).casefold()]
    terms=_item_terms(message)
    if not terms:
        return []
    scored=[(len(terms & _item_terms(entry["name"])),entry) for entry in entries]
    best=max((score for score,_entry in scored),default=0)
    return [entry for score,entry in scored if score==best and score>0]


def _get_orders_record(tool):
    return {"name":"get_orders","args":{},"status":"success"}


def _return_detail_turn(customer_id: str, history: list[Any], message: str, pending: dict | None):
    """Resolve customer-described items from their account without guessing an order ID or reason."""
    if pending:
        return None
    last_answer=_last_assistant_text(history)
    if is_clear_decline(message) and _RETURN_OFFER.search(last_answer):
        return "Okay, I won't prepare a return request. No return request has been submitted.","answer",[],[],False,None
    selected=_selected_return_context(history)
    reason=_return_reason(message) or _return_reason_in_history(history)
    accepted_offer=is_clear_confirmation(message) and bool(_RETURN_OFFER.search(last_answer))
    explicit_action=allows_new_proposal(message,None) or _return_action_in_history(history) or accepted_offer

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
            item=matches[0]
            selected={"name":item["name"],"order_id":item["order_id"],"item_id":item["item_id"]}
        else:
            if matches:
                choices="; ".join(f"{x['name']} (order {x['order_id']}, item {x['item_id']})" for x in matches)
                answer=f"You'd like to proceed with a return. I found these matching items: {choices}. Which one do you mean?"
            else:
                choices="; ".join(f"{x['name']} (order {x['order_id']}, item {x['item_id']})" for x in entries)
                answer=(f"You'd like to proceed with a return. Your account items are: {choices}. Which one do you mean?"
                        if choices else "You'd like to proceed with a return. Which product do you mean?")
            if not reason:
                answer += " " + _RETURN_REASON_QUESTION
            return answer,"clarify",offer_lookup_calls,[],False,None

    # A selected item and a stated reason are sufficient for Python eligibility.
    if selected and reason:
        tools=build_tools(customer_id,lambda _:None,allow_proposals=explicit_action)
        check_tool,create_tool=tools[2],tools[3]
        args={"order_id":selected["order_id"],"item_ids":[selected["item_id"]],"reason":reason}
        checked=check_tool.invoke(args)
        records=[{"name":"check_return_eligibility","args":args,
                  "status":"error" if checked.get("error_code") else "success"}]
        if not checked.get("eligible"):
            return checked.get("message","That item is not eligible for return."),"answer",offer_lookup_calls+records,[],False,None
        if explicit_action:
            proposal_holder={}
            tools=build_tools(customer_id,lambda proposal:proposal_holder.update(value=proposal),allow_proposals=True)
            result=tools[3].invoke(args)
            records.append({"name":"create_return_request","args":args,
                            "status":"pending_confirmation" if result.get("pending_confirmation") else "error"})
            if not result.get("pending_confirmation"):
                return result.get("message","I couldn't prepare that return."),"answer",records,[],False,None
            answer=f"I can create the return for {selected['name']} (order {selected['order_id']}, item {selected['item_id']}) for reason {reason}. Please reply yes to confirm."
            return answer,"answer",offer_lookup_calls+records,[],True,proposal_holder.get("value")
        answer=f"I found your {selected['name']} (order {selected['order_id']}, item {selected['item_id']}). It is eligible for return for reason {reason}. Would you like me to prepare the return?"
        return answer,"answer",offer_lookup_calls+records,[],False,None

    # A previous clarification carries the selected database item through the
    # ordinary session history. If the reason is still missing, ask for it.
    if selected and not reason and _return_reason_in_history(history) is None:
        prefix=(f"You'd like to proceed with a return for {selected['name']} (order {selected['order_id']}, item {selected['item_id']}). "
                if explicit_action else f"I found your {selected['name']} (order {selected['order_id']}, item {selected['item_id']}). ")
        return prefix+_RETURN_REASON_QUESTION,"clarify",offer_lookup_calls,[],False,None

    broad_reference=bool(_RETURN_BROAD_REFERENCE.search(message))
    specific_reference=bool(re.search(r"\b(?:return|send)\s+(?:my|the|this|that)\s+[a-z0-9][a-z0-9 -]{0,50}",message,re.I)
                            or re.search(r"\b(?:can|could|may)\s+i\s+(?:return|send)\s+(?:my|the|this|that)\s+[a-z0-9]",message,re.I))
    return_operation=bool(re.search(r"\b(?:return|send.{0,12}back)\b",message,re.I))
    general_return_reference=(return_operation and not re.search(r"\b(?:policy|window|period|days|how long|request status|tracking)\b",message,re.I))
    answering_return_clarification=bool(_RETURN_CLARIFICATION_CONTEXT.search(last_answer))
    is_return_request=broad_reference or specific_reference or general_return_reference or answering_return_clarification
    options_were_shown=last_answer.startswith(("I found these items on your account:", "I found these matching items:", "I couldn't find a matching item", "You'd like to proceed with a return."))
    if not (options_were_shown or is_return_request):
        return None

    get_orders,entries=_account_items(customer_id)
    get_orders_call=_get_orders_record(get_orders)
    if broad_reference:
        choices=[f"{entry['name']} (order {entry['order_id']}, item {entry['item_id']})" for entry in entries]
        if choices:
            answer="I found these items on your account: "+"; ".join(choices)
            answer += ". Which item do you mean? " + _RETURN_REASON_QUESTION
        else:
            answer="I couldn't find any order items on your account. Which product do you mean? " + _RETURN_REASON_QUESTION
        return answer,"clarify",[get_orders_call],[],False,None

    matches=_matching_account_items(message,entries)
    unique={(entry["order_id"],entry["item_id"]):entry for entry in matches}
    matches=list(unique.values())
    if len(matches)!=1:
        if matches:
            choices="; ".join(f"{entry['name']} (order {entry['order_id']}, item {entry['item_id']})" for entry in matches)
            prefix="You'd like to proceed with a return. " if explicit_action else ""
            answer=f"{prefix}I found these matching items: {choices}. Which one do you mean?"
        else:
            choices="; ".join(f"{entry['name']} (order {entry['order_id']}, item {entry['item_id']})" for entry in entries)
            prefix="You'd like to proceed with a return. " if explicit_action else ""
            answer=(f"{prefix}I couldn't find a matching item in your orders. Your account items are: {choices}. Which one do you mean?"
                    if choices else f"{prefix}I couldn't find an order item matching that description. Which product do you mean?")
        if not reason:
            answer += " " + _RETURN_REASON_QUESTION
        return answer,"clarify",[get_orders_call],[],False,None

    entry=matches[0]
    selected_answer=f"I found your {entry['name']} (order {entry['order_id']}, item {entry['item_id']})."
    if not reason:
        if explicit_action:
            selected_answer=f"You'd like to proceed with a return for {entry['name']} (order {entry['order_id']}, item {entry['item_id']})."
        return selected_answer+" "+_RETURN_REASON_QUESTION,"clarify",[get_orders_call],[],False,None
    # Re-enter the shared Python eligibility path with an explicit selected ID.
    context_message=AIMessage(content=selected_answer)
    result=_return_detail_turn(customer_id,history+[context_message],message,None)
    if result is None:
        return None
    answer,route,calls,citations,pending_value,proposal=result
    return answer,route,[get_orders_call]+calls,citations,pending_value,proposal


def _build_graph(customer_id: str, model, proposal_holder: dict, allow_proposals: bool):
    tools = build_tools(customer_id, lambda p: proposal_holder.update(value=p), allow_proposals)
    tool_by_name = {t.name: t for t in tools}

    def agent_node(state: AgentState):
        if state.get("tool_limit_reached"):
            return {"messages": [AIMessage(content=(
                "I couldn't check all matching items in one request. Which item do you mean, "
                "and is the reason that you changed your mind or that it is defective?"
            ))], "call_count": state.get("call_count", 0)}
        ai = model.bind_tools(tools).invoke(state["messages"])
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
        order_not_found = state.get("order_not_found", False)
        policy_uncovered = state.get("policy_uncovered", False)
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
                        policy_uncovered = policy_uncovered or not result.get("covered", False)
                    else:
                        non_policy_tool_used = True
                    if result.get("error_code") == "ORDER_NOT_FOUND":
                        order_not_found = True
                    status = "pending_confirmation" if result.get("pending_confirmation") else ("error" if result.get("error_code") or result.get("error") else "success")
                except Exception as exc:
                    result, status = {"error": "TOOL_ERROR", "message": str(exc)}, "error"
            records.append({"name": name, "args": args, "status": status})
            response_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {"messages": response_messages, "tool_calls": records, "citations": citations,
                "proposal": proposal_holder.get("value"), "order_not_found": order_not_found,
                "policy_uncovered": policy_uncovered, "non_policy_tool_used": non_policy_tool_used,
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
    """Useful graceful behavior with no API key; never guesses account data."""
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
            return "I couldn't find that order on your account.", "no_answer", records, [], False
        order = result["orders"][0]
        ship = order.get("shipment") or {}
        answer = f"Order {order_id} is {order['status'].replace('_', ' ')}."
        if ship.get("eta"):
            answer += f" Current estimated delivery date: {ship['eta']}."
        if ship.get("tracking_no"):
            answer += f" Carrier: {ship['carrier']}; tracking number: {ship['tracking_no']}."
        return answer, "answer", records, [], False
    if re.search(r"return|refund|warranty|shipping|cancel|exchange|policy|delivery", message, re.I):
        result = search_policy(message)
        records.append({"name": "search_policy", "args": {"query": message}, "status": "success" if result.get("covered") else "error"})
        if result.get("covered"):
            excerpt = result["excerpts"][0]
            return excerpt["text"], "answer", records, result["citations"][:1], False
        return "ShopSphere's policies do not cover that question. Please email the support team for help.", "no_answer", records, [], False
    if re.search(r"\b(hi|hello|hey|thanks|thank you)\b", message, re.I):
        return "Hello! I can help with ShopSphere orders, returns, and policies. What would you like to know?", "answer", records, [], False
    return "I can only help with ShopSphere orders and support policies. What would you like to know?", "out_of_scope", records, [], False


def run_chat(customer_id: str, history: list[Any], message: str, pending: dict | None):
    message = add_default_year(message)
    previous_proposal = pending
    if pending:
        if is_clear_confirmation(message):
            result = execute_proposal(customer_id, pending)
            name = "create_return_request" if pending["kind"] == "return" else "create_support_ticket"
            args = {k: v for k, v in pending.items() if k != "kind"}
            ok = bool(result.get("created"))
            record = {"name": name, "args": args, "status": "success" if ok else "error"}
            if ok:
                label = "return request" if pending["kind"] == "return" else "support ticket"
                id_value = result.get("request_id") or result.get("ticket_id")
                return f"Your {label} has been created. Reference: {id_value}.", "answer", [record], [], False, None
            return result.get("message", "I couldn't complete that request. Please contact support."), "no_answer", [record], [], False, None
        if is_clear_decline(message):
            label = "return request" if pending["kind"] == "return" else "support ticket"
            return f"Okay, I won't submit the {label}.", "answer", [], [], False, None
        if pending["kind"] == "return":
            revised = _revise_pending_return(customer_id, pending, message)
            if revised is not None:
                return revised
        # Pending proposals are discarded by every non-affirmative next message.
        pending = None

    if re.fullmatch(r"(?:just\s+)?do it[.!\s]*", message, re.I):
        return "Please tell me which ShopSphere action you want me to prepare.", "clarify", [], [], False, None

    # Resolve a missing order identifier before the LLM can choose an arbitrary
    # result from the optional unfiltered get_orders listing. The customer may
    # see their own matching order IDs so they can identify the intended one.
    if not re.search(r"\bSS-\d+\b", message, re.I) and _ORDER_LOOKUP_INTENT.search(message):
        get_orders = build_tools(customer_id, lambda _: None, allow_proposals=False)[0]
        result = get_orders.invoke({})
        orders = result.get("orders", [])
        ids = [order["order_id"] for order in orders]
        if ids:
            answer = f"I found these orders on your account: {', '.join(ids)}. Which one should I check?"
        else:
            answer = "I don't see an order on your account. Which order number should I check?"
        call = {"name": "get_orders", "args": {}, "status": "success"}
        return answer, "clarify", [call], [], False, None

    return_detail = _return_detail_turn(customer_id, history, message, pending)
    if return_detail is not None:
        return return_detail

    settings = get_settings()
    if settings.llm_provider != "openai" or not settings.openai_api_key:
        answer, route, records, citations, waiting = _fallback(message, customer_id)
        return answer, route, records, citations, waiting, None

    proposal_holder: dict = {}
    model = ChatOpenAI(model=settings.openai_model, temperature=0, api_key=settings.openai_api_key,
                       reasoning_effort="none", timeout=10, max_retries=0)
    compiled, tools = _build_graph(customer_id, model, proposal_holder, allows_new_proposal(message, previous_proposal))
    messages = [SystemMessage(content=SYSTEM)] + history[-12:] + [HumanMessage(content=message)]
    try:
        result = compiled.invoke({"messages": messages, "tool_calls": [], "citations": [], "call_count": 0},
                                 config={"recursion_limit": 18})
        last = result["messages"][-1]
        answer = _message_text(last.content)
        answer = _remove_inline_policy_citations(answer)
        proposal = proposal_holder.get("value")
        if result.get("order_not_found"):
            answer = "I couldn't find that order on your account."
            return answer, "no_answer", result.get("tool_calls", []), [], False, None
        if result.get("policy_uncovered"):
            suffix = "ShopSphere's policies do not cover that question. Please email the support team for help."
            if not result.get("non_policy_tool_used"):
                answer = suffix
            elif suffix.casefold() not in answer.casefold():
                answer = f"{answer} {suffix}".strip()
            return answer, "no_answer", result.get("tool_calls", []), [], bool(proposal), proposal
        if proposal and not re.search(r"\b(confirm|reply yes|say yes|proceed)\b", answer, re.I):
            answer = (answer + " Please reply yes if you'd like me to proceed.").strip()
        citations = result.get("citations", [])
        answer_lower = answer.casefold()
        cited = {(c["doc_id"], c["section"]) for c in citations
                 if c["doc_id"].casefold() in answer_lower and c["section"].casefold() in answer_lower}
        if cited:
            citations = [c for c in citations if (c["doc_id"], c["section"]) in cited]
        route = "clarify" if any(x in answer.lower() for x in ("which order", "what is the order", "which item", "what item")) else "answer"
        if "current policy does not cover" in answer.lower() or "cannot find that in current policy" in answer.lower():
            route = "no_answer"
        lower_answer = answer.lower()
        if "only help with shopsphere" in lower_answer or ("outside my scope" in lower_answer and "shopsphere" in lower_answer):
            route = "out_of_scope"
            citations = []
            result["tool_calls"] = []
            proposal = None
        if not answer:
            answer = "I couldn't find enough information to answer that. Please contact ShopSphere support."
            route = "no_answer"
        # A pending action must be accompanied by a confirmation request.
        if proposal:
            pending = proposal
        return answer, route, result.get("tool_calls", []), citations, bool(proposal), pending
    except Exception as exc:
        return f"I couldn't complete that request right now. Please try again or contact ShopSphere support. ({type(exc).__name__})", "no_answer", [], [], False, None
