"""Crawl official IRCC pages into visa_buddy_data/corpus/docs.jsonl.

    python -m pipeline.corpus

Breadth-first from START_URL and program hub pages, restricted to www.canada.ca,
robots.txt, and the program path prefixes in config. Each saved doc carries url,
title, fetched_at, program and text.
"""
import json
from collections import Counter, deque

from . import tracking
from .config import (CORPUS_DIR, EXTRA_URLS, IRCC_PREFIX, MAX_DEPTH, MAX_PAGES, MAX_PAGES_PER_PROGRAM,
                     MIN_DOC_CHARS, SEED_PATHS, START_URL)
from .fetch import fetch, program_for

DOCS = CORPUS_DIR / "docs.jsonl"


def tag(url: str) -> str | None:
    return EXTRA_URLS.get(url) or program_for(url)


def crawl() -> list[dict]:
    base = f"https://www.canada.ca{IRCC_PREFIX}"
    queue = deque([(START_URL, 0)] + [(base + p, 0) for p in SEED_PATHS] + [(u, 0) for u in EXTRA_URLS])
    seen, docs, per_prog, skipped = set(), {}, Counter(), Counter()
    while queue and len(docs) < MAX_PAGES:
        url, depth = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        prog = tag(url)
        if url != START_URL and prog is None:
            continue
        if prog and per_prog[prog] >= MAX_PAGES_PER_PROGRAM:
            skipped["program_cap"] += 1
            continue
        try:
            d = fetch(url)
        except Exception as e:
            print("ERR", url, e)
            skipped["error"] += 1
            continue
        if not d or d["status"] != 200:
            skipped["blocked_or_status"] += 1
            continue
        seen.add(d["url"])
        final_prog = tag(d["url"])
        if depth < MAX_DEPTH:
            for link in d["links"]:
                if link not in seen and tag(link):
                    queue.append((link, depth + 1))
        if final_prog is None or d["url"] in docs:
            continue  # hub outside programs, or a redirect to a page we already have
        if len(d["text"]) < MIN_DOC_CHARS:
            skipped["too_short"] += 1
            continue
        docs[d["url"]] = {"url": d["url"], "title": d["title"], "fetched_at": d["fetched_at"],
                          "program": final_prog, "text": d["text"]}
        per_prog[final_prog] += 1
        print(f"[{len(docs):>3}] {final_prog:<13} {len(d['text']):>6}  {d['url'].split('/services/')[-1]}")
    return list(docs.values()), skipped


def add_gold_sources():
    """Fetch gold source pages the crawl didn't reach. They're flagged, and the count is
    reported, so recall isn't credited to a crawler that never found the page."""
    from .config import GOLD_V1
    docs = [json.loads(l) for l in DOCS.read_text().splitlines() if l.strip()]
    have = {d["url"] for d in docs}
    gold = [json.loads(l) for l in GOLD_V1.read_text().splitlines() if l.strip()]
    need = sorted({u for g in gold for u in g["source_urls"]} - have)
    added = []
    for u in need:
        d = fetch(u)
        if d and d["status"] == 200 and d["url"] == u:
            added.append({"url": u, "title": d["title"], "fetched_at": d["fetched_at"], "program": tag(u),
                          "text": d["text"], "added_for_gold_coverage": True})
    with open(DOCS, "w") as f:
        for d in sorted(docs + added, key=lambda d: d["url"]):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"gold sources reached by crawl: {len(set().union(*[g['source_urls'] for g in gold])) - len(need)}; added: {[a['url'] for a in added]}")
    tracking.record("add_gold_sources", inputs=[GOLD_V1], outputs=[DOCS],
                    metrics={"gold_source_pages": len({u for g in gold for u in g["source_urls"]}),
                             "missing_after_crawl": len(need), "added": len(added),
                             "added_urls": [a["url"] for a in added]})


def main():
    import sys
    if "--add-gold-sources" in sys.argv:
        return add_gold_sources()
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    docs, skipped = crawl()
    with open(DOCS, "w") as f:
        for d in sorted(docs, key=lambda d: d["url"]):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    by_prog = Counter(d["program"] for d in docs)
    chars = Counter()
    for d in docs:
        chars[d["program"]] += len(d["text"])
    print(dict(by_prog), dict(skipped))
    tracking.record("crawl", outputs=[DOCS],
                    config={"max_pages": MAX_PAGES, "max_per_program": MAX_PAGES_PER_PROGRAM,
                            "max_depth": MAX_DEPTH, "min_doc_chars": MIN_DOC_CHARS, "seeds": SEED_PATHS},
                    metrics={"docs": len(docs), "docs_per_program": dict(by_prog),
                             "chars_per_program": dict(chars), "skipped": dict(skipped)})


if __name__ == "__main__":
    main()
