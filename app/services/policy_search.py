import json
import re
from pathlib import Path

from app.core.config import get_settings

STOPWORDS = {
    "what", "which", "where", "when", "why", "how", "much", "many", "are", "is", "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "can", "i", "do", "does", "you", "your", "my", "me", "this", "that", "it", "please", "tell", "about", "will", "would", "there", "be", "get", "from", "with", "standard", "policy", "policies", "shopsphere", "shop", "store", "device", "devices"
}


def _term(word: str) -> str:
    word = word.lower()
    if word in {"returning", "returned", "returns"}:
        return "return"
    if word in {"long", "take", "takes", "time", "timing", "day", "days"}:
        return "time"
    if word in {"track", "tracking"}:
        return "track"
    if word in {"fee", "fees", "cost", "costs", "price", "prices", "charge", "charges"}:
        return "cost"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and len(word) > 4:
        return word[:-1]
    return word


def _decode_policy(text: str) -> str:
    # Repair only common double-encoding; leave correctly decoded UTF-8 untouched.
    return (text.replace("\u00c2\u00a7", "\u00a7")
                .replace("\u00c2\u00b7", "\u00b7")
                .replace("\u00e2\u20ac\u201c", "\u2013")
                .replace("\u00e2\u201a\u00b9", "\u20b9"))


def _documents(text: str):
    for block in re.split(r"(?=^# )", text, flags=re.M):
        if not block.strip():
            continue
        match = re.search(r"\*\*Document ID:\*\*\s*([^ \u00b7]+)", block)
        if not match:
            continue
        doc_id = match.group(1)
        section = ""
        heading_title = ""
        paragraphs = []
        for line in block.splitlines():
            heading = re.match(r"^#{2,4}\s*\u00a7([\d.]+)\s+(.+)", line)
            if heading:
                section = "\u00a7" + heading.group(1)
                heading_title = heading.group(2).strip()
            elif line.strip() and not line.startswith("#"):
                body = f"{heading_title}: {line.strip()}" if section and heading_title else line.strip()
                paragraphs.append((section, body))
        yield doc_id, paragraphs


def search_policy(query: str):
    settings = get_settings()
    try:
        manifest = json.loads(Path(settings.manifest_path).read_text(encoding="utf-8"))
        policy_text = _decode_policy(Path(settings.policies_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": "POLICY_SOURCE_UNAVAILABLE", "message": str(exc), "citations": [], "excerpts": []}

    documents = dict(_documents(policy_text))
    today = settings.app_today
    future_intent = bool(re.search(r"\b(after|from|starting|effective|upcoming|future|change|new|next)\b|202[6-9]", query, re.I))
    valid = {
        entry["doc_id"]: entry for entry in manifest
        if entry["doc_id"] in documents and entry["status"] == "current"
        and (entry["effective_date"] <= today or future_intent)
    }
    terms = {
        _term(word) for word in re.findall(r"[\w]+", query)
        if len(word) > 2 and word.lower() not in STOPWORDS
    }
    topic_docs = {
        "shipping": "shipping_policy", "return": "returns_policy",
        "warranty": "warranty_policy", "refund": "refunds_policy",
        "cancel": "order_cancellation", "cancellation": "order_cancellation",
        "defective": "damaged_or_defective_items", "defect": "damaged_or_defective_items",
        "damage": "damaged_or_defective_items", "exchange": "exchanges",
    }
    preferred = {doc_id for term, doc_id in topic_docs.items() if term in terms}
    if "shipping" in terms:
        preferred.add("shipping_policy")
    if "track" in terms:
        preferred.add("faq_general")
    if future_intent and "shipping_policy" in preferred:
        preferred.add("shipping_policy_update")
    # Once the dated fee notice takes effect, ordinary fee questions must use
    # its changed terms instead of the older fee paragraph still in the base
    # shipping document. Preserve the base document for tracking and other
    # shipping topics that the notice does not replace.
    fee_intent = bool(terms & {"cost", "fee", "charge", "threshold", "free"})
    active_update = next((entry for entry in manifest
                         if entry.get("doc_id") == "shipping_policy_update"
                         and entry.get("status") == "current"
                         and entry.get("effective_date", "9999-12-31") <= today), None)
    if "shipping" in terms and fee_intent and active_update:
        preferred.discard("shipping_policy")
        preferred.add("shipping_policy_update")

    scored = []
    for doc_id, paragraphs in documents.items():
        doc = valid.get(doc_id)
        if not doc:
            continue
        for section, paragraph in paragraphs:
            if not section:
                continue
            words = {_term(word) for word in re.findall(r"[\w]+", paragraph.lower())}
            score = len(terms & words)
            heading = paragraph.split(":", 1)[0].lower()
            heading_terms = {_term(word) for word in re.findall(r"[\w]+", heading)}
            heading_matches = len(terms & heading_terms)
            score += 3 if heading_matches > 1 else heading_matches
            if score and doc_id in preferred:
                score += 3 if doc_id == "faq_general" and "track" in terms else 2
            if score:
                if doc.get("doc_type") == "faq":
                    score -= 0.25
                scored.append((score, doc_id, section, paragraph))
    if preferred:
        scored = [entry for entry in scored if entry[1] in preferred]
    scored.sort(key=lambda entry: -entry[0])
    if scored:
        best = scored[0][0]
        floor = max(1, best - 1) if best > 1 else 1
        scored = [entry for entry in scored if entry[0] >= floor]

    excerpts, citations, seen = [], [], set()
    for _score, doc_id, section, paragraph in scored:
        if (doc_id, section) in seen:
            continue
        seen.add((doc_id, section))
        doc = valid[doc_id]
        is_future = doc["effective_date"] > today
        if is_future:
            prefix = f"Future policy effective {doc['effective_date']}: "
        elif future_intent:
            prefix = f"Current policy as of {today}: "
        else:
            prefix = ""
        excerpts.append({
            "doc_id": doc_id, "section": section, "effective_date": doc["effective_date"],
            "future": is_future, "text": prefix + paragraph,
        })
        citation = {"doc_id": doc_id, "section": section}
        if is_future:
            citation["effective_date"] = doc["effective_date"]
        citations.append(citation)
        if len(excerpts) == 4:
            break
    if not excerpts:
        return {"covered": False, "citations": [], "excerpts": []}
    if any(citation["doc_id"] != "faq_general" for citation in citations):
        excerpts = [item for item in excerpts if item["doc_id"] != "faq_general"]
        citations = [item for item in citations if item["doc_id"] != "faq_general"]
    return {"covered": bool(excerpts), "citations": citations, "excerpts": excerpts}
