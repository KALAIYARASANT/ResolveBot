import re


_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_MONTH_FIRST = re.compile(rf"(?<!\w)(?P<month>{_MONTH})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?!\s*,?\s+\d{{4}})\b", re.I)
_DAY_FIRST = re.compile(rf"(?<!\w)(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH})(?!\s*,?\s+\d{{4}})\b", re.I)


def add_default_year(message: str) -> str:
    """Make month-name dates without a year explicit as 2026 before interpretation."""
    message = _MONTH_FIRST.sub(lambda m: f"{m.group(0)} 2026", message)
    return _DAY_FIRST.sub(lambda m: f"{m.group(0)} 2026", message)
