"""Chunk docs into ~1,000-character pieces with overlap, and split by page URL.

    python -m pipeline.chunk

Every chunk keeps url, title, program and its section heading. Pages cited by any
gold question (any gold version) are marked split="gold_heldout": they stay in the retrieval index
(eval and serving) but may never be a source for training targets. The split is
by page URL, so overlapping chunks of one page can't land on both sides.
"""
import hashlib
import re
import json
from collections import Counter

from . import tracking
from .config import CHUNK_CHARS, CHUNK_OVERLAP, CORPUS_DIR, GOLD_FILES

DOCS = CORPUS_DIR / "docs.jsonl"
CHUNKS = CORPUS_DIR / "chunks.jsonl"
SPLIT = CORPUS_DIR / "split.json"


def sections(text: str):
    """Yield (heading, body) using the '## ' markers fetch.py writes."""
    heading, buf = "", []
    for line in text.splitlines():
        if line.startswith("## "):
            if buf:
                yield heading, "\n".join(buf)
            heading, buf = line[3:].strip(), []
        elif line.strip():
            buf.append(line)
    if buf:
        yield heading, "\n".join(buf)


def pack(lines: list[str], size: int, overlap: int) -> list[str]:
    """Greedy line packing to ~size chars; next chunk starts with the last ~overlap chars of lines."""
    out, cur = [], []
    for line in lines:
        while len(line) > size:  # hard-wrap very long lines
            lines_head, line = line[:size], line[size - overlap:]
            if cur:
                out.append("\n".join(cur))
                cur = []
            out.append(lines_head)
        if cur and len("\n".join(cur + [line])) > size:
            out.append("\n".join(cur))
            tail, n = [], 0
            for l in reversed(cur):
                if n + len(l) > overlap:
                    break
                tail.insert(0, l)
                n += len(l) + 1
            cur = tail
        cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return out


def chunk_doc(doc: dict) -> list[dict]:
    # Merge small sections forward so headings like "On this page" don't become tiny chunks.
    blocks, pending, seen_sections = [], [], set()
    for heading, body in sections(doc["text"]):
        if heading.lower() == "on this page" or (heading, body) in seen_sections:
            continue  # questionnaire pages repeat the same section once per hidden branch
        seen_sections.add((heading, body))
        pending.append(f"{heading}\n{body}" if heading else body)
        # Flush once half a chunk has accumulated: short sections ("Fees: $7") get merged with
        # their neighbours instead of becoming chunks too small to retrieve on their own.
        if sum(len(p) for p in pending) >= CHUNK_CHARS * 0.5:
            blocks.append("\n".join(pending))
            pending = []
    if pending:
        if blocks and sum(len(p) for p in pending) < CHUNK_CHARS * 0.3:  # tiny tail: attach to the previous block
            blocks[-1] += "\n" + "\n".join(pending)
        else:
            blocks.append("\n".join(pending))
    out = []
    for block in blocks:
        heading = block.split("\n", 1)[0][:120]
        for piece in pack(block.split("\n"), CHUNK_CHARS, CHUNK_OVERLAP):
            out.append({"heading": heading, "text": piece})
    chunks = []
    for i, c in enumerate(out):
        # Stable id (same page + position -> same id) so predictions and SFT rows can point at chunks.
        cid = hashlib.sha1(f"{doc['url']}#{i}".encode()).hexdigest()[:12]
        chunks.append({"chunk_id": cid, "url": doc["url"], "title": doc["title"], "program": doc["program"],
                       "section": c["heading"], "position": i, "text": c["text"]})
    return chunks


# v4 page labels. Depth isn't what lets bad sources in (the crawl is fenced to IRCC's program
# pages); the risk is official pages that are wrong for an applicant asking today.
#   archived / historical: rules for past years or retired tools -> not indexed at all
#   employer / pilot: correct but for another audience or program -> indexed with a label,
#   never a source for training answers
def page_type(d: dict) -> str:
    url, title = d["url"], d["title"].lower()
    if "archived" in title:
        return "archived"
    if re.search(r"act-changes/(rules|requirements)-\d{4}", url):
        return "historical"
    if "/hire-" in url:
        return "employer"
    if re.search(r"pilot", url):
        return "pilot"
    return "current"


def main(docs_path=DOCS, out_dir=CORPUS_DIR, label_pages: bool = False):
    from pathlib import Path
    out_dir = Path(out_dir)
    chunks_path, split_path = out_dir / "chunks.jsonl", out_dir / "split.json"
    docs = [json.loads(l) for l in Path(docs_path).read_text().splitlines() if l.strip()]
    types = Counter()
    if label_pages:
        for d in docs:
            d["page_type"] = page_type(d)
            types[d["page_type"]] += 1
        docs = [d for d in docs if d["page_type"] not in ("archived", "historical")]
    gold_files = [p for p in GOLD_FILES if p.exists()]
    gold = [json.loads(l) for p in gold_files for l in p.read_text().splitlines() if l.strip()]
    gold_urls = {u for g in gold for u in g["source_urls"]}
    corpus_urls = {d["url"] for d in docs}
    missing = sorted(gold_urls - corpus_urls)
    if missing:  # a gold question whose source page isn't indexed could never be answered: stop
        raise SystemExit(f"gold source pages not in corpus (run pipeline.corpus --add-gold-sources):\n" + "\n".join(missing))

    chunks, seen_text, dropped = [], set(), 0
    for d in docs:
        split = "gold_heldout" if d["url"] in gold_urls else (
            "train_eligible" if d.get("page_type", "current") == "current" else "index_only")
        for c in chunk_doc(d):
            key = " ".join(c["text"].split())
            # Identical text already indexed: shared page blocks and questionnaire pages (one page
            # once produced 317 chunks) would otherwise fill all 5 retrieval slots with copies.
            if key in seen_text:
                dropped += 1
                continue
            seen_text.add(key)
            c["split"] = split
            if "page_type" in d:
                c["page_type"] = d["page_type"]
            chunks.append(c)
    with open(chunks_path, "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    index_only = {d["url"] for d in docs if d.get("page_type", "current") != "current"} - gold_urls
    split = {"gold_heldout": sorted(gold_urls),
             "train_eligible": sorted({d["url"] for d in docs} - gold_urls - index_only),
             "index_only": sorted(index_only)}
    assert not set(split["gold_heldout"]) & set(split["train_eligible"])  # no page on both sides
    split_path.write_text(json.dumps(split, indent=1))

    lens = sorted(len(c["text"]) for c in chunks)
    by_prog = Counter(c["program"] for c in chunks)
    by_split = Counter(c["split"] for c in chunks)
    metrics = {"page_types": dict(types), "docs": len(docs), "chunks": len(chunks), "duplicate_chunks_dropped": dropped, "chunks_per_program": dict(by_prog),
               "chunks_per_split": dict(by_split),
               "pages_per_split": {k: len(v) for k, v in split.items()},
               "chunk_chars_p10_p50_p90": [lens[len(lens) // 10], lens[len(lens) // 2], lens[9 * len(lens) // 10]],
               "chunk_len_hist": hist(lens)}
    print(json.dumps({k: v for k, v in metrics.items() if k != "chunk_len_hist"}, indent=1))
    tracking.record("chunk" if out_dir == CORPUS_DIR else f"chunk:{out_dir.name}", inputs=[Path(docs_path), *gold_files],
                    outputs=[chunks_path, split_path],
                    config={"chunk_chars": CHUNK_CHARS, "overlap": CHUNK_OVERLAP, "split_by": "page_url",
                            "dedupe": "repeated sections within page + identical chunk text across corpus"},
                    metrics=metrics)


def hist(lens, width=100):
    h = Counter(min(l // width, 15) * width for l in lens)
    return {str(k): h[k] for k in sorted(h)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:  # python -m pipeline.chunk visa_buddy_data/corpus_v2  (v4: with page labels)
        from pathlib import Path
        main(Path(sys.argv[1]) / "docs.jsonl", Path(sys.argv[1]), label_pages=True)
    else:
        main()
