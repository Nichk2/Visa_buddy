"""Chunk docs into ~1,000-character pieces with overlap, and split by page URL.

    python -m pipeline.chunk

Every chunk keeps url, title, program and its section heading. Pages cited by any
gold question (any gold version) are marked split="gold_heldout": they stay in the retrieval index
(eval and serving) but may never be a source for training targets. The split is
by page URL, so overlapping chunks of one page can't land on both sides.
"""
import hashlib
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
        if sum(len(p) for p in pending) >= CHUNK_CHARS * 0.5:
            blocks.append("\n".join(pending))
            pending = []
    if pending:
        if blocks and sum(len(p) for p in pending) < CHUNK_CHARS * 0.3:
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
        cid = hashlib.sha1(f"{doc['url']}#{i}".encode()).hexdigest()[:12]
        chunks.append({"chunk_id": cid, "url": doc["url"], "title": doc["title"], "program": doc["program"],
                       "section": c["heading"], "position": i, "text": c["text"]})
    return chunks


def main():
    docs = [json.loads(l) for l in DOCS.read_text().splitlines() if l.strip()]
    gold_files = [p for p in GOLD_FILES if p.exists()]
    gold = [json.loads(l) for p in gold_files for l in p.read_text().splitlines() if l.strip()]
    gold_urls = {u for g in gold for u in g["source_urls"]}
    corpus_urls = {d["url"] for d in docs}
    missing = sorted(gold_urls - corpus_urls)
    if missing:
        raise SystemExit(f"gold source pages not in corpus (run pipeline.corpus --add-gold-sources):\n" + "\n".join(missing))

    chunks, seen_text, dropped = [], set(), 0
    for d in docs:
        split = "gold_heldout" if d["url"] in gold_urls else "train_eligible"
        for c in chunk_doc(d):
            key = " ".join(c["text"].split())
            if key in seen_text:  # identical text already indexed (shared page blocks)
                dropped += 1
                continue
            seen_text.add(key)
            c["split"] = split
            chunks.append(c)
    with open(CHUNKS, "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    split = {"gold_heldout": sorted(gold_urls),
             "train_eligible": sorted(corpus_urls - gold_urls)}
    assert not set(split["gold_heldout"]) & set(split["train_eligible"])
    SPLIT.write_text(json.dumps(split, indent=1))

    lens = sorted(len(c["text"]) for c in chunks)
    by_prog = Counter(c["program"] for c in chunks)
    by_split = Counter(c["split"] for c in chunks)
    metrics = {"docs": len(docs), "chunks": len(chunks), "duplicate_chunks_dropped": dropped, "chunks_per_program": dict(by_prog),
               "chunks_per_split": dict(by_split),
               "pages_per_split": {k: len(v) for k, v in split.items()},
               "chunk_chars_p10_p50_p90": [lens[len(lens) // 10], lens[len(lens) // 2], lens[9 * len(lens) // 10]],
               "chunk_len_hist": hist(lens)}
    print(json.dumps({k: v for k, v in metrics.items() if k != "chunk_len_hist"}, indent=1))
    tracking.record("chunk", inputs=[DOCS, *gold_files], outputs=[CHUNKS, SPLIT],
                    config={"chunk_chars": CHUNK_CHARS, "overlap": CHUNK_OVERLAP, "split_by": "page_url",
                            "dedupe": "repeated sections within page + identical chunk text across corpus"},
                    metrics=metrics)


def hist(lens, width=100):
    h = Counter(min(l // width, 15) * width for l in lens)
    return {str(k): h[k] for k in sorted(h)}


if __name__ == "__main__":
    main()
