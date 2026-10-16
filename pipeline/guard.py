"""Serving-time guard against answers that flip a rule stated in the sources.

v3 still says "you don't have to stop working" where the source says "You must stop working"
(safety t04) and "you can't get a 3-year PGWP" where the source says you can (gold pg06): the
base model's prior knowledge overrides the sources, and more training data didn't fix it.

After generation, each answer sentence is compared with the source sentence it matches best
(pipeline.text.polarity_flips). If one is negated and the other isn't, the answer is kept and
a visible note quotes the official sentence(s) with their citation, so the reader can see the
mismatch. A first version replaced the whole answer with the quote; measured on v3's saved
answers it fixed pg06 but swapped a correct ee08 answer for an irrelevant quote (the check
paired the wrong sentence), so it now never removes an answer. Measured with
`python -m pipeline.guard`: fires on pg06 (real), ee08 and wp07 (false alarms), misses t04.
"""
import json
import re
import sys
from pathlib import Path

from .prompt import REFUSAL
from .text import canon_lines, norm, polarity_flips


NOTE = "Note: part of this answer may not match the official wording. The source says:"


def _source_sentence(canon_sentence: str, hits: list[dict]) -> tuple[str, dict] | None:
    """Find the original (un-canonicalized) source sentence and the hit it came from. If it
    stops mid-thought ("..., as long as" followed by a bullet list), the next lines are included
    so the quote isn't cut before its conditions."""
    for h in hits:
        lines = h["text"].splitlines()
        for i, line in enumerate(lines):
            for sent in re.split(r"(?<=[.!?])\s+", line):
                c = canon_lines(sent).strip()
                if c and (c in canon_sentence or canon_sentence in c):
                    out = sent.strip()
                    j = i + 1
                    while not re.search(r"[.!?]\s*$", out) and j < len(lines) and j <= i + 4:
                        out += "\n  " + lines[j].strip()
                        j += 1
                    return out, h
    return None


def guard(answer: str, hits: list[dict]) -> tuple[str, list[dict]]:
    """Return (answer to show, flags). Refusals pass through untouched."""
    if norm(REFUSAL) in norm(answer):
        return answer, []
    flips = polarity_flips(answer, "\n".join(h["text"] for h in hits))
    if not flips:
        return answer, []
    quotes, seen = [], set()
    for _, src in flips:
        found = _source_sentence(src, hits)
        if found and found[0] not in seen:
            seen.add(found[0])
            quotes.append(f"\"{found[0]}\" (Source: {found[1]['title']}, {found[1]['url']})")
    flags = [{"answer": a, "source": s} for a, s in flips]
    if not quotes:  # couldn't locate the original wording: keep the answer, flag only
        return answer, flags
    return f"{answer}\n\n{NOTE}\n" + "\n".join(f"- {q}" for q in quotes), flags


def evaluate(tags=(("v3", "gold_v2"), ("safety_v3", "safety_v1"))):
    """Apply the guard to saved answers and re-grade them (grader v3). No regeneration."""
    from . import tracking
    from .config import EVAL_DIR, TOP_K
    from .eval_answers import summarize
    from .grade import grade
    from .retriever import EMB, META, Retriever

    ret = Retriever()
    summary = {}
    for tag, gold_name in tags:
        gold = {json.loads(l)["id"]: json.loads(l) for l in (EVAL_DIR / f"{gold_name}.jsonl").read_text().splitlines()}
        src = EVAL_DIR / f"{tag}_predictions.jsonl"
        rows = []
        for p in map(json.loads, src.read_text().splitlines()):
            hits = ret.search(p["question"], TOP_K)
            assert [h["url"] for h in hits] == [x["url"] for x in p["retrieved"]], f"{p['id']}: retrieval differs"
            shown, flags = guard(p["answer"], hits)
            rows.append({**p, "answer": shown, "answer_before_guard": p["answer"], "guard_flags": flags,
                         "grade": grade(shown, hits, gold[p["id"]])})
        name = f"{tag}_guard"
        (EVAL_DIR / f"{name}_predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        rep = {"tag": name, "gold_file": f"{gold_name}.jsonl", "grader_version": 3, "guarded_from": src.name,
               "guard_triggered": [r["id"] for r in rows if r["guard_flags"]], **summarize(rows)}
        (EVAL_DIR / f"{name}_report.json").write_text(json.dumps(rep, indent=1))
        summary[name] = {"triggered": rep["guard_triggered"], **{k: rep["overall"][k] for k in
                         ("fact_coverage", "contradicts_source", "refusal_correct_on_must_refuse", "over_refusal_rate")}}
        tracking.record(f"guard_eval:{tag}", inputs=[src, EVAL_DIR / f"{gold_name}.jsonl", EMB, META, Path(__file__)],
                        outputs=[EVAL_DIR / f"{name}_predictions.jsonl", EVAL_DIR / f"{name}_report.json"],
                        config={"guard": "polarity_flips -> keep answer, append note quoting source sentences",
                                "grader_version": 3},
                        metrics={"overall": rep["overall"], "triggered": rep["guard_triggered"]})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    evaluate()
