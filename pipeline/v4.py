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
import json
import re
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


def _sentences(text):
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-").strip()
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if len(s.strip()) > 15]
    return out


def verify_quotes(raw_quotes, hits):
    """Map each model quote to the real source sentence it came from (verbatim from canada.ca).
    Quotes that don't match any sentence of the retrieved pages closely enough are dropped."""
    kept, seen = [], set()
    for q in raw_quotes or []:
        if not isinstance(q, dict) or not str(q.get("text", "")).strip():
            continue
        qt = norm(str(q["text"]))
        best, score = None, 0.0
        for h in hits:
            for s in _sentences(h["text"]):
                ns = norm(s)
                r = 1.0 if qt in ns or ns in qt and len(ns) > 0.6 * len(qt) else SequenceMatcher(None, qt, ns).ratio()
                if r > score:
                    best, score = (s, h), r
        if best and score >= MIN_QUOTE_MATCH and best[0] not in seen:
            seen.add(best[0])
            kept.append({"text": best[0], "title": best[1]["title"], "url": best[1]["url"], "match": round(score, 3)})
    return kept


def verify_answer(answer, quotes):
    """Return the list of problems; empty means the answer can be shown."""
    qtext = "\n".join(q["text"] for q in quotes)
    problems = [f"number not in quotes: {n}" for n in _numbers(URL_RE.sub(" ", answer)) if not _number_in(n, norm(qtext))]
    problems += [f"polarity flip: {a!r} vs {s!r}" for a, s in polarity_flips(answer, qtext, flexible=True)]
    return problems


def render(answer, quotes, status):
    cites = "\n".join(f'- "{q["text"]}" (Source: {q["title"]}, {q["url"]})' for q in quotes)
    if status == "answered":
        return f"{answer.strip()}\n\nOfficial wording:\n{cites}"
    return f"I can't summarize this safely, but here is what the official sources say:\n{cites}"


def run_batch(questions, hits_list, chat, max_tokens=(400, 200)):
    """Run the pipeline for many questions at once. `chat(list_of_conversations, max_tokens)`
    returns one text per conversation (batched: one vLLM call per step on Modal)."""
    ext_raw = chat([extract_messages(q, h) for q, h in zip(questions, hits_list)], max_tokens[0])
    results, to_answer = [], []
    for i, (q, hits, raw) in enumerate(zip(questions, hits_list, ext_raw)):
        obj = _parse_json(raw) or {}
        r = {"question": q, "extract_raw": raw, "quotes": [], "answer_raw": None, "problems": []}
        if obj.get("refuse") is True:
            r.update(status="refused_policy", response=REFUSAL_OUTPUT)
        else:
            r["quotes"] = verify_quotes(obj.get("quotes"), hits)
            if not r["quotes"]:
                r.update(status="refused_no_quotes", response=REFUSAL_OUTPUT)
            else:
                to_answer.append(i)
        results.append(r)
    if to_answer:
        ans_raw = chat([answer_messages(questions[i], results[i]["quotes"]) for i in to_answer], max_tokens[1])
        for i, raw in zip(to_answer, ans_raw):
            r = results[i]
            r["answer_raw"] = raw
            r["problems"] = verify_answer(raw, r["quotes"])
            r["status"] = "answered" if not r["problems"] else "quotes_only"
            r["response"] = render(raw, r["quotes"], r["status"])
    return results
