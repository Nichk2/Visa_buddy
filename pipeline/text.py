"""Text normalization and fact matching shared by gold validation, retrieval eval and grading."""
import re

_SUBS = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
_STOP = set("a an the of to for in on at and or is are be you your it its if as by with from that this can".split())


def norm(s: str) -> str:
    for k, v in _SUBS.items():
        s = s.replace(k, v)
    return " ".join(s.lower().split())


def tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9$][a-z0-9$,.'/-]*[a-z0-9]|[a-z0-9$]", norm(s)) if t not in _STOP]


def fact_in(fact: str, text: str, min_token_frac: float = 0.8) -> bool:
    """True if `fact` is supported by `text`: exact normalized substring, or at least
    min_token_frac of its content tokens present. Numbers and money must match exactly."""
    f, t = norm(fact), norm(text)
    if f in t:
        return True
    ft = tokens(fact)
    if not ft:
        return False
    tt = set(tokens(text))
    numeric = [x for x in ft if re.search(r"\d", x)]
    if any(x not in tt and x.rstrip(".,") not in t for x in numeric):
        return False
    return sum(x in tt for x in ft) / len(ft) >= min_token_frac
