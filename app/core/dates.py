import re


_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_MONTH_FIRST = re.compile(rf"(?<!\w)(?P<month>{_MONTH})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?!\s*,?\s+\d{{4}})\b", re.I)
_DAY_FIRST = re.compile(rf"(?<!\w)(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH})(?!\s*,?\s+\d{{4}})\b", re.I)


def add_default_year(message: str) -> str:
    """Make month-name dates without a year explicit as 2026 before interpretation."""
    message = _MONTH_FIRST.sub(lambda m: f"{m.group(0)} 2026", message)
    return _DAY_FIRST.sub(lambda m: f"{m.group(0)} 2026", message)


_ISO_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_NAMED_DATE = re.compile(rf"(?<!\w)(?:\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?),?\s+\d{{4}}\b", re.I)


def find_date(message: str) -> str | None:
    """The first calendar date written in the message (2025-08-18, 18 August 2025, August 18, 2025) as YYYY-MM-DD."""
    from dateutil import parser
    found = min((m for m in (_ISO_DATE.search(message), _NAMED_DATE.search(message)) if m), key=lambda m: m.start(), default=None)
    if not found:
        return None
    try:
        return parser.parse(found.group(0)).date().isoformat()
    except (ValueError, OverflowError):
        return None
