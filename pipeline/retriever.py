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
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def build():
    chunks = [json.loads(l) for l in CHUNKS.read_text().splitlines() if l.strip()]
    model = _model()
    emb = model.encode([embed_text(c) for c in chunks], batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True).astype(np.float32)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB, emb)
    META.write_text(CHUNKS.read_text())
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

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        q = self.model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        scores = self.emb @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i]), "rank": r + 1} for r, i in enumerate(top)]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for h in Retriever().search(" ".join(sys.argv[1:])):
            print(f"{h['rank']} {h['score']:.3f} [{h['program']}] {h['title']} :: {h['section']}\n   {h['url']}")
    else:
        build()
