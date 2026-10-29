import json
import math
import re
from pathlib import Path

from app.core.config import get_settings

STOPWORDS = {
    "what", "which", "where", "when", "why", "how", "much", "many", "are", "is", "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "can", "i", "do", "does", "you", "your", "my", "me", "this", "that", "it", "please", "tell", "about", "will", "would", "there", "be", "get", "from", "with", "standard", "policy", "policies", "shopsphere", "shop", "store", "device", "devices",
    "have", "has", "had", "something", "any", "know", "want", "need", "could", "should", "if", "am", "was", "were", "not", "no", "yes", "so", "but", "at", "by", "as", "can't", "cannot",
}
# Plain word groups (not policy facts): different ways of asking the same thing.
_SYNONYMS = {
    **dict.fromkeys(("long", "take", "takes", "time", "timing", "day", "days"), "time"),
    **dict.fromkeys(("fee", "fees", "cost", "costs", "charge", "charges"), "cost"),
}
MAX_EXCERPTS = 5
MAX_PER_DOCUMENT = 3  # keeps one busy document from crowding out the FAQ or a neighbouring policy


def _term(word: str) -> str:
    """Lower-case stem: shipping/shipped/ships -> ship, statuses -> status, change/changing -> chang."""
    word = word.lower()
    if word in _SYNONYMS:
        return _SYNONYMS[word]
    for suffix in ("ing", "ed", "age", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3 and not word.endswith(("ss", "us", "is")):
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "ls":
        word = word[:-1]
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def significant_terms(text: str) -> set[str]:
    return {_term(word) for word in re.findall(r"\w+", text) if len(word) > 2 and word.lower() not in STOPWORDS}


def _decode_policy(text: str) -> str:
    # Repair only common double-encoding; leave correctly decoded UTF-8 untouched.
    return (text.replace("Â§", "§")
                .replace("Â·", "·")
                .replace("â€“", "–")
                .replace("â‚¹", "₹"))


def _documents(text: str):
    """Yield (doc_id, sections); a section is (number, heading, body) with the body's lines joined."""
    for block in re.split(r"(?=^# )", text, flags=re.M):
        if not block.strip():
            continue
        match = re.search(r"\*\*Document ID:\*\*\s*([^ ·]+)", block)
        if not match:
            continue
        sections, current, parent = [], None, ""
        for line in block.splitlines():
            heading = re.match(r"^(#{2,4})\s*§([\d.]+)\s+(.+)", line)
            if heading:
                title = heading.group(3).strip()
                if heading.group(1) == "##":
                    parent = title  # FAQ questions are level-2 headings; keep them with their answer
                current = {"section": "§" + heading.group(2),
                           "heading": title if heading.group(1) == "##" else (f"{parent} / {title}" if parent else title),
                           "lines": []}
                sections.append(current)
            elif line.strip() and not line.startswith("#") and current is not None:
                current["lines"].append(line.strip())
        yield match.group(1), [(s["section"], s["heading"], "\n".join(s["lines"])) for s in sections if s["lines"]]


_QUERY_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?:\s+\d{1,2}(?:st|nd|rd|th)?,?)?\s+\d{4}\b|\b20\d{2}\b", re.I)


def _names_later_date(query: str, today: str) -> bool:
    """True when the query names a date after the business date (today's own date is not a future policy)."""
    from dateutil import parser
    for match in _QUERY_DATE.finditer(query):
        try:  # a month and year alone means the first of that month; a bare year, 1 January
            when = parser.parse(match.group(0), default=parser.parse("2000-01-01")).date().isoformat()
        except (ValueError, OverflowError):
            continue
        if when > today:
            return True
    return False


def search_policy(query: str):
    settings = get_settings()
    try:
        manifest = json.loads(Path(settings.manifest_path).read_text(encoding="utf-8"))
        policy_text = _decode_policy(Path(settings.policies_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"error": "POLICY_SOURCE_UNAVAILABLE", "message": str(exc), "citations": [], "excerpts": []}

    documents = dict(_documents(policy_text))
    today = settings.app_today
    # A document that has not started yet is considered only when the question is about the future.
    future_intent = bool(re.search(r"\b(after|from|starting|effective|upcoming|future|next)\b", query, re.I)) or _names_later_date(query, today)
    valid = {
        entry["doc_id"]: entry for entry in manifest
        if entry["doc_id"] in documents and entry["status"] == "current"
        and (entry["effective_date"] <= today or future_intent)
    }
    terms = significant_terms(query)
    if not terms:
        return {"covered": False, "citations": [], "excerpts": []}
    # Most of a short question must appear in the paragraph (so one shared word is not coverage); long,
    # model-written queries only need two shared terms.
    needed = min(2, max(1, math.ceil(len(terms) / 2)))

    scored = []
    for doc_id, sections in documents.items():
        doc = valid.get(doc_id)
        if not doc:
            continue
        # The document title says what a section is about ("Warranty Policy"); FAQ titles are boilerplate.
        title_terms = set() if doc.get("doc_type") == "faq" else significant_terms(doc.get("title", ""))
        for section, heading, body in sections:
            heading_terms, body_terms = significant_terms(heading), significant_terms(body)
            matched = terms & (heading_terms | body_terms | title_terms)
            if len(matched) < needed:
                continue
            score = len(matched) + 0.5 * len(terms & heading_terms)
            # Ties go to policy over FAQ (the FAQ defers to the policy), then to the newer document.
            order = (-score, doc.get("doc_type") == "faq", -int(doc["effective_date"].replace("-", "")))
            same_topic = (frozenset(matched), frozenset(terms & heading_terms))
            scored.append((order, doc_id, section, f"{heading}: {body}", same_topic))
    scored.sort(key=lambda entry: entry[0])
    # Keep sections within one point of the best match, separately for documents in force and future ones, so a
    # strongly matching future notice cannot push out the policy that applies today.
    best = {}
    for entry in scored:
        group = valid[entry[1]]["effective_date"] > today
        best[group] = max(best.get(group, 0), -entry[0][0])
    # A question can ask for two facts that sit side by side in one notice (a fee and a threshold); a sibling section
    # under the same heading in the same document as a kept match stays even if it scores a little lower.
    def parent(entry):
        return entry[1], entry[3].split(" / ", 1)[0] if " / " in entry[3].split(": ", 1)[0] else None
    kept = [e for e in scored if -e[0][0] >= max(best[valid[e[1]]["effective_date"] > today] - 1, 0)]
    families = {parent(e) for e in kept if parent(e)[1]}
    scored = [e for e in scored if e in kept or parent(e) in families]

    # Two in-force policy sections about the same topic (same heading terms, same matched terms) can disagree
    # when a newer document changes an older one; the newer document wins. FAQs and future documents are exempt.
    def in_force_policy(doc_id):
        return valid[doc_id].get("doc_type") != "faq" and valid[doc_id]["effective_date"] <= today
    newest = {}
    for _order, doc_id, _section, _paragraph, topic in scored:
        if topic[1] and in_force_policy(doc_id):
            newest[topic] = max(newest.get(topic, ""), valid[doc_id]["effective_date"])
    scored = [e for e in scored if not (e[4][1] and in_force_policy(e[1]) and valid[e[1]]["effective_date"] < newest[e[4]])]

    # A newer in-force policy document that names an older one (an update notice) replaces the older sections whose
    # heading it repeats with more words ("Standard shipping fee" -> "New standard shipping fee"), even when the
    # replacement was not itself retrieved. Sections it does not restate (for example Plus membership) stay.
    def superseded(doc_id, heading):
        old_terms = significant_terms(heading)
        if len(old_terms) < 2 or not in_force_policy(doc_id):
            return False
        for newer_id, newer in valid.items():
            if (newer_id == doc_id or not in_force_policy(newer_id) or newer["effective_date"] <= valid[doc_id]["effective_date"]
                    or not any(doc_id in body for _s, _h, body in documents[newer_id])):
                continue
            if any(old_terms <= significant_terms(newer_heading) for _s, newer_heading, _b in documents[newer_id]):
                return True
        return False
    scored = [e for e in scored if not superseded(e[1], e[3].split(": ", 1)[0])]

    excerpts, per_document = [], {}
    for _order, doc_id, section, paragraph, _topic in scored:
        if per_document.get(doc_id, 0) >= MAX_PER_DOCUMENT:
            continue
        per_document[doc_id] = per_document.get(doc_id, 0) + 1
        doc = valid[doc_id]
        is_future = doc["effective_date"] > today
        if is_future:
            prefix = f"Future policy effective {doc['effective_date']}: "
        elif future_intent:
            prefix = f"Current policy as of {today}: "
        else:
            prefix = ""
        excerpts.append({
            "doc_id": doc_id, "section": section, "doc_type": doc.get("doc_type"),
            "effective_date": doc["effective_date"], "future": is_future, "text": prefix + paragraph,
        })
        if len(excerpts) == MAX_EXCERPTS:
            break
    citations = [{"doc_id": x["doc_id"], "section": x["section"],
                  **({"effective_date": x["effective_date"]} if x["future"] else {})} for x in excerpts]
    return {"covered": bool(excerpts), "citations": citations, "excerpts": excerpts}
