"""The one retriever. Dataset building, evaluation and the chat server all call `Retriever.search`.

    python -m pipeline.retriever            # build the index from corpus/chunks.jsonl
    python -m pipeline.retriever "query"    # debug search
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


def embed_text(c: dict) -> str:
    # Title and section heading make short chunks findable ("Fees" alone says nothing).
    return f"{c['title']} | {c['section']}\n{c['text']}"


def _model():
    from sentence_transformers import SentenceTransformer
    # CPU on purpose: identical vectors on the Mac, Colab and the server, so a query retrieves
    # the same top-5 everywhere (the Colab runs are checked against a local re-run).
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def build():
    chunks = [json.loads(l) for l in CHUNKS.read_text().splitlines() if l.strip()]
    model = _model()
    emb = model.encode([embed_text(c) for c in chunks], batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True).astype(np.float32)  # unit vectors: dot product = cosine
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB, emb)
    META.write_text(CHUNKS.read_text())  # metadata saved next to the vectors so row i always matches vector i
    INFO.write_text(json.dumps({"embed_model": EMBED_MODEL, "chunks": len(chunks), "dim": int(emb.shape[1]),
                                "normalized": True, "similarity": "cosine (dot product)"}, indent=1))
    tracking.record("index", inputs=[CHUNKS], outputs=[EMB, META, INFO],
                    config={"embed_model": EMBED_MODEL, "embed_text": "title | section + text"},
                    metrics={"chunks": len(chunks), "dim": int(emb.shape[1])})


class Retriever:
    def __init__(self):
        self.chunks = [json.loads(l) for l in META.read_text().splitlines() if l.strip()]
        self.emb = np.load(EMB)
        assert len(self.chunks) == len(self.emb), "index and chunk metadata out of sync"
        self.model = _model()

    def search(self, query: str, k: int = TOP_K, allow=None) -> list[dict]:
        """Top-k chunks by cosine similarity. `allow(chunk) -> bool` restricts the pool
        (dataset building uses it to draw contexts from training pages only)."""
        q = self.model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        scores = self.emb @ q  # brute force over ~3.6k chunks is milliseconds; no vector DB needed
        out = []
        for i in np.argsort(-scores):
            if allow is None or allow(self.chunks[i]):
                out.append({**self.chunks[i], "score": float(scores[i]), "rank": len(out) + 1})
                if len(out) == k:
                    break
        return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for h in Retriever().search(" ".join(sys.argv[1:])):
            print(f"{h['rank']} {h['score']:.3f} [{h['program']}] {h['title']} :: {h['section']}\n   {h['url']}")
    else:
        build()
