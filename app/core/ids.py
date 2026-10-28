import re

# "SS10512", "SS 10512", "ss-10512" and "SS10458-1" all name the same order or item.
_ORDER_OR_ITEM_ID = re.compile(r"\bSS[\s-]?(\d+)(?:-(\d+))?\b", re.I)


def normalize_order_ids(text: str) -> str:
    """Rewrite every order/item reference in free text to canonical SS-<digits>[-<n>] form."""
    return _ORDER_OR_ITEM_ID.sub(lambda m: f"SS-{m.group(1)}" + (f"-{m.group(2)}" if m.group(2) else ""), text)


def normalize_id(value):
    """Canonical form of a single order/item ID; other values pass through unchanged.

    Normalizing never grants access: every lookup still filters by the authenticated customer.
    """
    return normalize_order_ids(value.strip()) if isinstance(value, str) else value
