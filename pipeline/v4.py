"""Visa Buddy v4 pipeline: retrieve -> extract quotes -> answer from quotes -> verify.

Why this shape (see reports/V4_PLAN.md): v1-v3 showed a model will state an outdated rule even
when the current one is in its sources (r03: "Yes" while citing the policy's 2022 end date).
So the facts shown to the user must be the official sentences themselves, checked by code:

1. extract: the model copies the sentence(s) that answer the question. Each quote is matched to
   a real sentence in the retrieved pages; anything that can't be matched is dropped, so the
   user only ever sees text that is on canada.ca.
2. answer: the model writes 1-3 sentences from those quotes only.
3. verify: numbers in the answer must appear in the quotes, and no answer sentence may flip the
   polarity of a quote. If either check fails, the user gets the quotes without the answer:
   stiff but never wrong. The answer is never shown before or without its quotes.

The model is a callable taking a batch of chat conversations and returning texts, so the same
code runs with vLLM on Modal (bake-off) and MLX on the Mac (the shipped app).
"""
import calendar
import json
import re
from datetime import date
from difflib import SequenceMatcher

from .grade import _number_in, _numbers, URL_RE
from .prompt import REFUSAL_OUTPUT
from .text import norm, polarity_flips

EXTRACT_SYSTEM = """You find evidence in official Canadian immigration (IRCC) sources.
Copy, word for word, the sentence or sentences from the numbered SOURCES that answer the QUESTION. Copy them exactly; do not rephrase or shorten them.
Reply with JSON only, in this form: {"quotes": [{"source": <source number>, "text": "<exact sentence>"}]}
If none of the sources answers the question, reply: {"quotes": []}
If the question asks how to misrepresent information or avoid detection, asks for legal strategy, or asks you to predict whether someone's application will succeed, reply: {"refuse": true}"""

ANSWER_SYSTEM = """You answer Canadian immigration questions using ONLY the official quotes provided.
- If it is a yes/no question, start your answer with "Yes" or "No".
- Say only what the quotes say. Never add rules, fees, dates, durations or conditions that are not in the quotes.
- The quotes are current official rules. If they differ from what you remember, the quotes are right.
- Answer in 1 to 3 short sentences. Do not include links or source names; they are shown separately."""

MIN_QUOTE_MATCH = 0.85  # similarity to a real source sentence needed to accept a quote


def _sources_block(hits):
    return "\n\n".join(f"[{i}] {h['title']}\n{h['text']}" for i, h in enumerate(hits, 1))


def extract_messages(question, hits):
    return [{"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": f"SOURCES:\n{_sources_block(hits)}\n\nQUESTION: {question}"}]


def answer_messages(question, quotes):
    block = "\n".join(f'[{i}] "{q["text"]}"' for i, q in enumerate(quotes, 1))
    return [{"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"QUOTES:\n{block}\n\nQUESTION: {question}"}]


def _parse_json(text):
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


MAX_WINDOW = 15  # a quote may span a lead-in plus a long bullet list (sp06 lists 5+ items)


def _units(text):
    """Candidate passages a quote can come from: each sentence, and runs of 1-8 consecutive
    lines. The first bake-off compared quotes with single lines only and rejected correct
    quotes that ran across a bullet list ("...managed through Express Entry: - CEC - FSWP")."""
    lines = [l.strip().lstrip("-").strip() for l in text.splitlines() if l.strip()]
    out = []
    for line in lines:
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if len(s.strip()) > 15]
    for n in range(2, MAX_WINDOW + 1):
        # Joined with line breaks, not spaces: merging bullets into one run-on sentence put the
        # "can't" of one bullet in the same clause as the next one and confused the polarity checks.
        out += ["\n".join(lines[i:i + n]) for i in range(len(lines) - n + 1)]
    return out


def _quote_text(q):
    """Models return quotes as {"text": ...} objects or as plain strings (Phi-4-mini did)."""
    if isinstance(q, str):
        return q
    if isinstance(q, dict):
        return str(q.get("text") or q.get("quote") or "")
    return ""


def verify_quotes(raw_quotes, hits):
    """Map each model quote to the real source passage it came from (verbatim from canada.ca).
    Quotes that don't match any passage of the retrieved pages closely enough are dropped.
    Among passages that match, the shortest wins, so the user sees no more than was quoted."""
    kept, seen = [], set()
    for q in raw_quotes or []:
        qt = norm(_quote_text(q))
        if len(qt) < 6:  # table rows are short ("1 $15,263" from the proof-of-funds table)
            continue
        best, score = None, 0.0
        for h in hits:
            for u in _units(h["text"]):
                nu = norm(u)
                if qt in nu and len(nu) <= 1.5 * len(qt) + 40:
                    r = 1.0
                elif nu in qt and len(nu) > 0.6 * len(qt):
                    r = 0.95
                else:
                    r = SequenceMatcher(None, qt, nu).ratio()
                if r > score or (r == score and best and len(u) < len(best[0])):
                    best, score = (u, h), r
        if best and score >= MIN_QUOTE_MATCH and best[0] not in seen and not _PLACEHOLDER.search(best[0]):
            seen.add(best[0])
            kept.append({"text": best[0], "title": best[1]["title"], "url": best[1]["url"], "match": round(score, 3)})
    return kept


# --- expired quotes -------------------------------------------------------------------
# A current page can still describe a policy that has ended. In the bake-off Qwen quoted
# "You can complete up to 100% of your studies online from outside Canada between March 2020
# and August 31, 2022, because of a temporary COVID-19 policy" and answered "Yes" (rules r03).
# A quote is expired when it describes a temporary policy/measure AND gives an end date before
# today. Rules about past *applications* ("if you applied before November 1, 2024, you're
# exempt") don't match: they carry no temporary-policy wording and still apply today.
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE = rf"(?:{_MONTHS})(?:\s+\d{{1,2}},?)?\s+20\d\d"
_TEMP = re.compile(r"\b(temporary|public policy|measures?|pandemic|covid)\b", re.I)
_END = re.compile(rf"(?:until|ended(?: on)?|expired(?: on)?|in place until|changed as of|and)\s+({_DATE})", re.I)


def _to_date(text):
    m = re.match(rf"({_MONTHS})(?:\s+(\d{{1,2}}),?)?\s+(20\d\d)", text.strip(), re.I)
    month = [x.lower() for x in _MONTHS.split("|")].index(m.group(1).lower()) + 1
    year = int(m.group(3))
    day = int(m.group(2)) if m.group(2) else calendar.monthrange(year, month)[1]
    return date(year, month, day)


_PLACEHOLDER = re.compile(r"error loading", re.I)  # canada.ca fills fees by JavaScript; our copy keeps the placeholder


def is_expired(text, today=None):
    today = today or date.today()
    for sent in re.split(r"(?<=[.!?])\s+|\n", text):
        if _TEMP.search(sent) and any(_to_date(m.group(1)) < today for m in _END.finditer(sent)):
            return True
    return False


def _expired_pages(hits, today=None):
    """Pages (URLs) among the retrieved sources that describe an ended temporary policy."""
    return {h["url"] for h in hits if is_expired(h["text"], today)}


def _in_expired_section(quote, url, expired_pages):
    """Page-level expiry: in bake-off run 3 the model dropped r03's dated quote and quoted
    "You're eligible for this temporary policy if..." from another chunk of the same page. On a
    page that describes an ended temporary policy, quotes that refer to a temporary policy or the
    pandemic belong to that policy too. (A chunk-level first version missed it: the two
    sentences were in different chunks.)"""
    return url in expired_pages and bool(_TEMP.search(quote))


def _without_expired(hits, today=None):
    """The same sources with expired lines removed (and, on pages with an ended temporary
    policy, the lines that refer to it), for the one retry."""
    exp = _expired_pages(hits, today)
    return [{**h, "text": "\n".join(l for l in h["text"].splitlines()
                                     if not is_expired(l, today) and not (h["url"] in exp and _TEMP.search(l)))}
            for h in hits]


_LEAD_YN = re.compile(r"^\W*(yes|no)\b[,.!]?\s*", re.I)
_NEG_WORDS = re.compile(r"\b(not|no|never|don't|doesn't|can't|cannot|isn't|aren't|won't|wouldn't|shouldn't|mustn't)\b", re.I)


def verdict_conflict(answer):
    """True when the opening Yes/No contradicts the clause right after it: "Yes, you do not have
    to show results" (teacher, rules r01) or "No, you can apply". Checked on the first clause
    only, after normalizing curly apostrophes; refined once after its first version flagged
    correct answers ("No. You can’t...", "No. You can only start...", "Yes, you may..., and this
    does not affect...")."""
    a = norm(answer)
    m = _LEAD_YN.match(a)
    if not m:
        return False
    first = re.split(r"(?<=[.!?])\s+", a[m.end():].strip(), maxsplit=1)[0]
    clause = re.split(r",|;|\b(?:and|but|because|as long as|if|unless|while|which)\b", first, maxsplit=1)[0]
    if not clause.strip():
        return False
    negated = len(_NEG_WORDS.findall(clause)) % 2 == 1
    if m.group(1) == "yes":
        return negated
    if negated or re.search(r"\bonly\b", clause):  # "No. You can only..." restricts: consistent
        return False
    return re.match(r"(you|they|he|she|we)\s+(can|may|are allowed to|are eligible)\b", clause.strip()) is not None


def verify_answer(answer, quotes):
    """Return the list of problems; empty means the answer can be shown."""
    qtext = "\n".join(q["text"] for q in quotes)
    problems = [f"number not in quotes: {n}" for n in _numbers(URL_RE.sub(" ", answer)) if not _number_in(n, norm(qtext))]
    problems += [f"polarity flip: {a!r} vs {s!r}" for a, s in polarity_flips(answer, qtext, flexible=True)]
    if verdict_conflict(answer):
        problems.append("yes/no contradicts its own sentence")
    return problems


def render(answer, quotes, status):
    cites = "\n".join('- "' + q["text"].replace("\n", "\n  ") + f'" (Source: {q["title"]}, {q["url"]})' for q in quotes)
    if status == "answered":
        return f"{answer.strip()}\n\nOfficial wording:\n{cites}"
    return f"I can't summarize this safely, but here is what the official sources say:\n{cites}"


def run_batch(questions, hits_list, chat, max_tokens=(400, 200), today=None):
    """Run the pipeline for many questions at once. `chat(list_of_conversations, max_tokens)`
    returns one text per conversation (batched: one vLLM call per step on Modal)."""
    ext_raw = chat([extract_messages(q, h) for q, h in zip(questions, hits_list)], max_tokens[0])
    results, retry = [], []
    for i, (q, hits, raw) in enumerate(zip(questions, hits_list, ext_raw)):
        obj = _parse_json(raw) or {}
        r = {"question": q, "extract_raw": raw, "quotes": [], "answer_raw": None, "problems": [], "expired_dropped": []}
        if obj.get("refuse") is True:
            r.update(status="refused_policy", response=REFUSAL_OUTPUT)
        else:
            quotes = verify_quotes(obj.get("quotes"), hits)
            exp = _expired_pages(hits, today)
            gone = lambda q_: is_expired(q_["text"], today) or _in_expired_section(q_["text"], q_["url"], exp)
            r["expired_dropped"] = [q_["text"] for q_ in quotes if gone(q_)]
            r["quotes"] = [q_ for q_ in quotes if not gone(q_)]
            if not r["quotes"] and r["expired_dropped"]:
                retry.append(i)  # only expired passages were quoted: try once without them
        results.append(r)
    if retry:
        clean = {i: _without_expired(hits_list[i], today) for i in retry}
        raw2 = chat([extract_messages(questions[i], clean[i]) for i in retry], max_tokens[0])
        for i, raw in zip(retry, raw2):
            obj = _parse_json(raw) or {}
            results[i]["extract_raw_retry"] = raw
            if obj.get("refuse") is True:
                results[i].update(status="refused_policy", response=REFUSAL_OUTPUT)
            else:
                exp = _expired_pages(hits_list[i], today)
                results[i]["quotes"] = [q_ for q_ in verify_quotes(obj.get("quotes"), clean[i])
                                        if not is_expired(q_["text"], today) and not _in_expired_section(q_["text"], q_["url"], exp)]
    to_answer = []
    for i, r in enumerate(results):
        if "status" in r:
            continue
        if r["quotes"]:
            to_answer.append(i)
        else:
            r.update(status="refused_no_quotes", response=REFUSAL_OUTPUT)
    if to_answer:
        ans_raw = chat([answer_messages(questions[i], results[i]["quotes"]) for i in to_answer], max_tokens[1])
        for i, raw in zip(to_answer, ans_raw):
            r = results[i]
            r["answer_raw"] = raw
            r["problems"] = verify_answer(raw, r["quotes"])
            r["status"] = "answered" if not r["problems"] else "quotes_only"
            r["response"] = render(raw, r["quotes"], r["status"])
    return results
