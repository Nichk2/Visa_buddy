"""The one retriever. Dataset building, evaluation and the chat server all call `Retriever.search`.

    python -m pipeline.retriever            # build the index from corpus/chunks.jsonl
    python -m pipeline.retriever "query"    # debug search

Two methods. "dense" (MiniLM cosine) is what v1-v3 used and stays the default so their results
reproduce. "hybrid" (v4) takes candidates from dense AND BM25 keyword search, then a small
cross-encoder re-ranks them: dense misses exact terms ("IMM 5476", "CLB 7"), BM25 misses
paraphrases, and the re-ranker reads question and chunk together instead of comparing vectors.
"""
import json
import sys

import numpy as np

from . import tracking
from .config import CORPUS_DIR, EMBED_MODEL, INDEX_DIR, TOP_K

CHUNKS = CORPUS_DIR / "chunks.jsonl"
EMB = INDEX_DIR / "embeddings.npy"
META = INDEX_DIR / "chunks.jsonl"
INFO = INDEX_DIR / "index_info.json"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~90 MB, CPU, fast enough for 40 pairs per query
CANDIDATES = 20  # per method before re-ranking


def embed_text(c: dict) -> str:
    # Title and section heading make short chunks findable ("Fees" alone says nothing).
    return f"{c['title']} | {c['section']}\n{c['text']}"


def _model():
    from sentence_transformers import SentenceTransformer
    # CPU on purpose: identical vectors on the Mac, Colab and the server, so a query retrieves
    # the same top-5 everywhere (the Colab runs are checked against a local re-run).
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def build(chunks_path=CHUNKS, index_dir=INDEX_DIR):
    from pathlib import Path
    index_dir = Path(index_dir)
    emb_p, meta_p, info_p = index_dir / "embeddings.npy", index_dir / "chunks.jsonl", index_dir / "index_info.json"
    chunks = [json.loads(l) for l in Path(chunks_path).read_text().splitlines() if l.strip()]
    model = _model()
    emb = model.encode([embed_text(c) for c in chunks], batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True).astype(np.float32)  # unit vectors: dot product = cosine
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_p, emb)
    meta_p.write_text(Path(chunks_path).read_text())  # metadata saved next to the vectors so row i always matches vector i
    info_p.write_text(json.dumps({"embed_model": EMBED_MODEL, "chunks": len(chunks), "dim": int(emb.shape[1]),
                                "normalized": True, "similarity": "cosine (dot product)"}, indent=1))
    tracking.record("index" if index_dir == INDEX_DIR else f"index:{index_dir.name}", inputs=[Path(chunks_path)],
                    outputs=[emb_p, meta_p, info_p],
                    config={"embed_model": EMBED_MODEL, "embed_text": "title | section + text"},
                    metrics={"chunks": len(chunks), "dim": int(emb.shape[1])})


class BM25:
    """Okapi BM25 over title + section + text. Small enough to build in memory at load time."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        from collections import Counter
        from .text import tokens
        self.tok = tokens
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=np.float32)
        self.avg = float(self.len.mean())
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: float(np.log(1 + (n - c + 0.5) / (c + 0.5))) for w, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tf), dtype=np.float32)
        for w in set(self.tok(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, t in enumerate(self.tf):
                f = t.get(w)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


class Retriever:
    def __init__(self, index_dir=INDEX_DIR, method: str = "dense"):
        from pathlib import Path
        index_dir = Path(index_dir)
        self.method = method
        self.chunks = [json.loads(l) for l in (index_dir / "chunks.jsonl").read_text().splitlines() if l.strip()]
        self.emb = np.load(index_dir / "embeddings.npy")
        assert len(self.chunks) == len(self.emb), "index and chunk metadata out of sync"
        self.model = _model()
        if method == "hybrid":
            from sentence_transformers import CrossEncoder
            self.bm25 = BM25([embed_text(c) for c in self.chunks])
            self.reranker = CrossEncoder(RERANK_MODEL, device="cpu")

    def _top(self, scores, n, allow):
        out = []
        for i in np.argsort(-scores):
            if allow is None or allow(self.chunks[i]):
                out.append(int(i))
                if len(out) == n:
                    break
        return out

    def search(self, query: str, k: int = TOP_K, allow=None) -> list[dict]:
        """Top-k chunks. `allow(chunk) -> bool` restricts the pool (dataset building uses it to
        draw contexts from training pages only)."""
        q = self.model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        dense = self.emb @ q  # brute force over a few thousand chunks is milliseconds; no vector DB needed
        if self.method == "dense":
            return [{**self.chunks[i], "score": float(dense[i]), "rank": r + 1}
                    for r, i in enumerate(self._top(dense, k, allow))]
        cand = list(dict.fromkeys(self._top(dense, CANDIDATES, allow) + self._top(self.bm25.scores(query), CANDIDATES, allow)))
        rr = self.reranker.predict([(query, embed_text(self.chunks[i])) for i in cand], show_progress_bar=False)
        order = [cand[j] for j in np.argsort(-np.asarray(rr))][:k]
        best = {cand[j]: float(rr[j]) for j in range(len(cand))}
        return [{**self.chunks[i], "score": best[i], "rank": r + 1} for r, i in enumerate(order)]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for h in Retriever().search(" ".join(sys.argv[1:])):
            print(f"{h['rank']} {h['score']:.3f} [{h['program']}] {h['title']} :: {h['section']}\n   {h['url']}")
    else:
        build()
