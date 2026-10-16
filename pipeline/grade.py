"""Deterministic grader. No LLM judge: it reads the answer, the retrieved chunks and the
gold row, and nothing else. (An LLM judge sharing the prompt being tuned would reward its
own style; a deterministic grader gives the same score on every rerun.)

- citation_hit: the answer contains a URL that is one of the retrieved chunk URLs
- fact_coverage: share of expected_facts present in the answer AND in the retrieved
  chunk text (a fact the model says but the chunks don't contain doesn't count)
- refused / refusal_correct: refused iff the answer contains the fixed refusal sentence;
  correct when refused == must_refuse
- unsupported: the answer states a checkable claim missing from the chunks. "Checkable"
  means numbers with their unit (money, durations, CLB levels, counts) and URLs. This is
  a proxy: a wrong claim in plain words with no number is not caught.
- answer_in_context: every expected fact is in the retrieved chunks. Lets the summary tell a
  wrong refusal (answer was there) from a correct abstention (retrieval missed it).

Grader v1 matched facts by wording close to the page; v2 matches gold_v2's accepted keys
after canonicalizing money and contractions (v1 scored "$7 CAD" vs "CAN$7" as a miss).
Both versions stay callable so earlier scores can be reproduced.
"""
import re

from .config import OFFICIAL_INDEX_URL
from .prompt import REFUSAL
from .text import fact_in, key_in, key_polarity_conflict, norm

URL_RE = re.compile(r"https?://[^\s)\]>,\"']+")
# A number plus an optional unit word, e.g. "$15,263", "24 hours", "180 days", "CLB 7", "1,095".
UNITS = r"%|hours?|days?|weeks?|months?|years?|points?|times?|people|members?"
NUM_RE = re.compile(rf"(?:\$\s?|CAN\$\s?|\bCLB\s|\bNCLC\s)?\d[\d,]*(?:\.\d+)?(?:\s?(?:{UNITS})\b)?", re.I)


def _urls(text):  # trailing "." from the end of a sentence isn't part of the URL
    return {u.rstrip(".") for u in URL_RE.findall(text)}


def _numbers(text):
    out = set()
    for m in NUM_RE.finditer(text):
        s = m.group(0).strip()
        if re.fullmatch(r"\[?\d\]?", s) or re.fullmatch(r"\d", s):  # source markers like [1] or bare digits
            continue
        out.add(s)
    return out


def _number_in(n: str, ctx_n: str) -> bool:
    """The number must appear in the chunks with the same unit: "10 points" needs
    "10 points" (or "10 point"), "$155" needs "$155", "CLB 7" needs "CLB 7". A bare
    number only needs the digits."""
    flat = ctx_n.replace(",", "")
    n = norm(n).replace(",", "")
    digits = re.sub(r"[^\d.]", "", n).strip(".")
    if not digits:
        return True
    unit = re.search(rf"({UNITS})$", n)
    if unit:
        stem = unit.group(1).rstrip("s") if unit.group(1) != "%" else "%"
        return re.search(rf"(?<![\d.]){re.escape(digits)}\s?{re.escape(stem)}", flat) is not None
    if n.startswith("$") or "can$" in n:
        return re.search(rf"\$\s?{re.escape(digits)}(?![\d])", flat) is not None
    if n.startswith(("clb", "nclc")):
        return re.search(rf"{n.split()[0]}\)?\s?(level\s)?{re.escape(digits)}(?![\d])", flat) is not None
    return re.search(rf"(?<![\d.]){re.escape(digits)}(?![\d])", flat) is not None


GRADER_VERSION = 3  # v3 = v2 + polarity check + titles read by the number check


def grade(answer: str, hits: list[dict], gold: dict, version: int = GRADER_VERSION) -> dict:
    """version=1 reproduces the original grading; version=2 matches each fact by its
    accepted keys (gold_v2 `fact_keys`) after money/contraction canonicalization;
    version=3 also refuses credit when the answer states a fact with the opposite
    polarity to the sources (and flags it as a contradiction)."""
    ctx = "\n".join(h["text"] for h in hits)
    # v3: source titles count as source text for numbers ("version 2016" lives in the title)
    num_ctx = "\n".join(f"{h.get('title', '')}\n{h['text']}" for h in hits) if version >= 3 else ctx
    retrieved = {h["url"] for h in hits}
    cited = _urls(answer)
    refused = norm(REFUSAL) in norm(answer)

    unsupported_claims = [u for u in cited if u not in retrieved and u.rstrip("/") != OFFICIAL_INDEX_URL.rstrip("/")]
    num_ctx_n = norm(num_ctx)
    for n in _numbers(URL_RE.sub(" ", answer)):
        if not _number_in(n, num_ctx_n):
            unsupported_claims.append(n)

    facts = gold.get("expected_facts", [])
    contradicted = []
    if version >= 2 and gold.get("fact_keys"):
        supported = []
        for f, keys in zip(facts, gold["fact_keys"]):
            matched = [k for k in keys if key_in(k, answer) and key_in(k, ctx)]
            if version >= 3 and any(key_polarity_conflict(k, answer, ctx) for k in matched):
                contradicted.append(f)  # answer states the fact the other way round
            elif matched:
                supported.append(f)
        in_ctx = all(any(key_in(k, ctx) for k in keys) for keys in gold["fact_keys"])
    else:
        supported = [f for f in facts if fact_in(f, answer) and fact_in(f, ctx)]
        in_ctx = all(fact_in(f, ctx) for f in facts)
    # Rules set: the answer's opening "Yes"/"No" against the correct verdict. Stating the
    # opposite rule (v3: "No, you can't get a 3-year PGWP") is the failure v4 targets.
    lead = re.match(r"\W*(yes|no)\b", norm(answer))
    verdict_given = lead.group(1) if lead and not refused else None
    return {
        "grader_version": version,
        "verdict_given": verdict_given,
        "wrong_verdict": bool(gold.get("verdict") and verdict_given and verdict_given != gold["verdict"]),
        "citation_hit": bool(cited & retrieved),
        "fact_coverage": (len(supported) / len(facts)) if facts else None,
        "facts_supported": supported,
        "facts_contradicted": contradicted,
        "contradicts_source": bool(contradicted),
        "refused": refused,
        # every expected fact is present in the retrieved sources (None for must-refuse rows)
        "answer_in_context": (in_ctx if facts else None),
        "refusal_correct": refused == gold["must_refuse"],
        "unsupported": bool(unsupported_claims),
        "unsupported_claims": sorted(set(unsupported_claims)),
    }
