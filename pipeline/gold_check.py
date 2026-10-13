"""Validate a gold file against the live official pages it cites.

    python -m pipeline.gold_check eval/gold_v1.jsonl

Checks: schema, counts per program, every expected_fact appears on at least one of
its source pages, and every not_on_page refusal marker is absent from its page.
This reads source pages directly, never the chunk index.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .config import PROGRAMS
from .fetch import fetch
from .text import fact_in, key_in

REQUIRED = {"id", "question", "program", "expected_facts", "source_urls", "must_refuse"}


def main(path: str) -> int:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    errors = []
    ids = Counter(r["id"] for r in rows)
    errors += [f"duplicate id {i}" for i, n in ids.items() if n > 1]
    for r in rows:
        missing = REQUIRED - r.keys()
        if missing:
            errors.append(f"{r.get('id')}: missing {missing}")
        if r["program"] not in PROGRAMS:
            errors.append(f"{r['id']}: unknown program {r['program']}")
        if not r["must_refuse"] and (not r["expected_facts"] or not r["source_urls"]):
            errors.append(f"{r['id']}: answerable row needs facts and sources")

    pages = {}
    for u in sorted({u for r in rows for u in r["source_urls"]}):
        d = fetch(u)
        if not d or d["status"] != 200:
            errors.append(f"source fetch failed: {u}")
            continue
        if d["url"] != u:
            errors.append(f"source redirects: {u} -> {d['url']} (use the final URL)")
        pages[u] = d["text"]

    for r in rows:
        texts = [pages.get(u, "") for u in r["source_urls"]]
        v1_texts = [pages.get(u, "") for u in r["source_urls"] if u not in r.get("extra_source_urls", [])]
        for fct in r["expected_facts"]:
            if not any(fact_in(fct, t) for t in v1_texts):
                errors.append(f"{r['id']}: fact not on its source pages: {fct!r}")
        for fct, keys in zip(r["expected_facts"], r.get("fact_keys", [])):
            if not any(key_in(k, t) for k in keys for t in texts):
                errors.append(f"{r['id']}: no accepted key for {fct!r} on its source pages")
        for u in r.get("extra_source_urls", []):  # an extra source must state every fact
            for fct, keys in zip(r["expected_facts"], r.get("fact_keys", [])):
                if not any(key_in(k, pages.get(u, "")) for k in keys):
                    errors.append(f"{r['id']}: extra source {u} lacks {fct!r}")
        if r["must_refuse"] and r.get("refusal_reason") == "not_on_page":
            for m in r.get("answer_markers", []):
                if any(re.search(m, t, re.I) for t in texts):
                    errors.append(f"{r['id']}: marker {m!r} matches its page, so the page does answer it")

    answerable = Counter(r["program"] for r in rows if not r["must_refuse"])
    refuse = sum(r["must_refuse"] for r in rows)
    print(f"rows={len(rows)} answerable={sum(answerable.values())} must_refuse={refuse}")
    for p in PROGRAMS[:-1]:
        print(f"  {p:<14} {answerable.get(p, 0)}")
        if answerable.get(p, 0) < 4:
            errors.append(f"program {p} has fewer than 4 answerable questions")
    if len(rows) < 40 or refuse < 8:
        errors.append("need >= 40 questions and >= 8 must_refuse")
    for e in errors:
        print("ERROR", e)
    print("OK" if not errors else f"{len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
