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


# --- grader v3: polarity (negation) check ---------------------------------------
# Key matching alone can't see negation: on safety t04 the source says "You must stop working"
# and the answer "No, you don't have to stop working" still matched the key "stop working".
# v3 compares the polarity of the clause around each key in the answer with the source.
# The rule was refined once after reading every flag from its first version (11 of 13 were
# false alarms): lines are kept apart, only negations before the key count, and conditions
# ("if ...") are their own clause. Each false-alarm pattern is now a unit test.
_NEG = re.compile(r"\b(no|not|never|nor|without|don't|doesn't|didn't|can't|isn't|aren't|won't|wasn't|weren't|"
                  r"shouldn't|mustn't|couldn't|wouldn't|neither|none|nobody|nothing)\b")
# Clause boundaries. In "You don't need proof of funds if you're applying under the CEC" the
# negation belongs to the main clause, not to the "if" clause that names the program.
_CLAUSE = re.compile(r"[.;:!?\n,]|\b(?:but|unless|except|however|if|when|while|because|whereas)\b")
# A leading "No," / "Yes," answers the question; it doesn't negate the fact that follows.
_LEAD = re.compile(r"^\s*(?:-\s*)?(?:yes|no)\b[,.!]?\s*")


def canon_lines(s: str) -> str:
    """canon() per line, keeping line breaks: list items and table rows stay separate clauses
    (flattening them let a "not" from one row negate the next)."""
    return "\n".join(canon(line) for line in s.splitlines())


def _prefixes(key_c: str, text_c: str) -> list[str]:
    """For each occurrence of the key: the text from the start of its clause up to the key.
    Negation usually comes before what it negates ("you can't [get a 3-year PGWP]"); a "not"
    after the key belongs to the next phrase ("lose your status and may not be approved")."""
    out = []
    for m in re.finditer(rf"(?<![\w$]){re.escape(key_c)}(?!\w)", text_c):
        starts = [b.end() for b in _CLAUSE.finditer(text_c, 0, m.start())]
        out.append(text_c[starts[-1] if starts else 0:m.start()])
    return out


def _polarity(prefix: str) -> int:
    """1 = affirmed, -1 = negated: parity of negation words before the key in its clause."""
    return -1 if len(_NEG.findall(_LEAD.sub("", prefix))) % 2 else 1


def key_polarity_conflict(key: str, answer: str, source: str) -> bool:
    """True if the answer uses `key` with a polarity that no source mention of `key` has."""
    k, a, s = canon(key), canon_lines(answer), canon_lines(source)
    src = {_polarity(p) for p in _prefixes(k, s)}
    ans = {_polarity(p) for p in _prefixes(k, a)}
    return bool(src) and bool(ans - src)


def _anchor(a_words: list[str], s_words: list[str]) -> str | None:
    """Longest run of consecutive words shared by two sentences, if it has >= 2 content words
    (e.g. "stop working"). It plays the role a gold fact key plays in grader v3."""
    best = ""
    for i in range(len(a_words)):
        for j in range(len(s_words)):
            k = 0
            while i + k < len(a_words) and j + k < len(s_words) and a_words[i + k] == s_words[j + k]:
                k += 1
            if k and len(" ".join(a_words[i:i + k])) > len(best):
                best = " ".join(a_words[i:i + k])
    return best if len(tokens(best)) >= 2 else None


def polarity_flips(answer: str, source: str, min_overlap: float = 0.5) -> list[tuple[str, str]]:
    """For training data (no fact keys): pair each answer sentence with the source sentence that
    shares most of its content words, take the longest phrase they share, and apply grader v3's
    clause rule to it (negation before the phrase, within its clause). A mismatch means the
    answer flipped a rule, like v2's "you don't have to stop working". Returns
    (answer sentence, source sentence) pairs."""
    def sentences(t):
        t = re.sub(r"\(source:[^)]*\)|https?://\S+", " ", canon_lines(t))
        return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n", t) if len(tokens(x)) >= 3]
    words = lambda x: re.findall(r"[a-z0-9$'][a-z0-9$,.'/-]*[a-z0-9]|[a-z0-9$]", x)
    src = [(x, set(tokens(x))) for x in sentences(source)]
    flips = []
    for a in sentences(answer):
        at = set(tokens(a))
        best, ov = None, 0.0
        for x, xt in src:
            o = len(at & xt) / len(at) if at else 0
            if o > ov:
                best, ov = x, o
        if not best or ov < min_overlap:
            continue
        anchor = _anchor(words(a), words(best))
        if anchor and key_polarity_conflict(anchor, a, best):
            flips.append((a, best))
    return flips
