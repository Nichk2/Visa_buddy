"""Text normalization and fact matching shared by gold validation, retrieval eval and grading."""
import re

# canada.ca uses curly quotes and non-breaking spaces; models answer with straight ones.
# Without this, "you can't" on the page never matches "you can't" in an answer.
_SUBS = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
# Function words carry no fact; ignoring them keeps token matching on content.
_STOP = set("a an the of to for in on at and or is are be you your it its if as by with from that this can".split())


def norm(s: str) -> str:
    for k, v in _SUBS.items():
        s = s.replace(k, v)
    return " ".join(s.lower().split())


def tokens(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9$][a-z0-9$,.'/-]*[a-z0-9]|[a-z0-9$]", norm(s)) if t not in _STOP]


def fact_in(fact: str, text: str, min_token_frac: float = 0.8) -> bool:
    """True if `fact` is supported by `text`: exact normalized substring, or at least
    min_token_frac of its content tokens present. Numbers and money must match exactly.
    This is grader v1's matcher (kept to reproduce v1 scores); its token fallback can be too
    loose ("CLB" and "7" anywhere), which is one reason gold_v2 switched to exact keys."""
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


# --- grader v2 canonicalization -------------------------------------------------
_CONTRACTIONS = [(r"\bcannot\b", "can't"), (r"\bcan not\b", "can't"), (r"\bdo not\b", "don't"),
                 (r"\bdoes not\b", "doesn't"), (r"\bis not\b", "isn't"), (r"\bare not\b", "aren't"),
                 (r"\bwill not\b", "won't"), (r"\bdid not\b", "didn't")]
# "$CAN 150", "CAN$7", "Can$100", "CAD $155", "$7 CAD", "$7 CAN" -> "$150", "$7", ...
_MONEY = re.compile(r"(?:\$\s?can\b|can\$|cad\s?\$|\$)\s?(\d[\d,]*(?:\.\d+)?)(?:\s?(?:cad|can)\b)?")


def canon(s: str) -> str:
    s = norm(s)
    for pat, rep in _CONTRACTIONS:
        s = re.sub(pat, rep, s)
    s = re.sub(r"\((clb|nclc|teer|noc)\)", r"\1", s)  # "benchmarks (clb) 7" -> "benchmarks clb 7"
    s = re.sub(r"\b(clb|nclc)\s+level\s+(\d)", r"\1 \2", s)  # "clb level 7" -> "clb 7"
    return _MONEY.sub(lambda m: "$" + m.group(1), s)


def key_in(key: str, text: str) -> bool:
    """Grader v2: an accepted key matches as an exact span after canonicalization."""
    k, t = canon(key), canon(text)
    return re.search(rf"(?<![\w$]){re.escape(k)}(?!\w)", t) is not None
