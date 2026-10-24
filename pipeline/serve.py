"""Visa Buddy chat API: the same retriever, model loading and checks as the eval.

    python -m pipeline.serve --pipeline v4 --adapter visa_buddy_data/models/v4   # Colab T4, the shipped v4
    python -m pipeline.serve --adapter visa_buddy_data/models/v3                 # v3 (lab phase), kept for comparison
    python -m pipeline.serve --model Qwen/Qwen2.5-0.5B-Instruct --smoke   # local plumbing test only

Endpoints, unchanged for the React app (src/services/visaBuddyApi.ts):
  POST /chat   {"message": str, "max_length": int}  ->  {"response": str}
  GET  /health ->  {"status": str, "model_loaded": bool}

model_loaded is true only when the model, the adapter and the retrieval index all loaded.
There is no fallback model: v1 silently served DialoGPT when Llama failed to load, and the
app looked like it worked. Here a failed load means /health says false and /chat returns 503.
"""
import argparse
import logging
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import TOP_K
from .guard import guard
from .prompt import build_messages
from .v4 import run_batch

log = logging.getLogger("visa_buddy")
MAX_NEW_TOKENS = 600  # cap regardless of the client's max_length: long answers drift from the sources

app = FastAPI(title="Visa Buddy API", version="2.0.0")
# The React app runs on another origin (localhost or a hosted frontend) and calls this API directly.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE = {"retriever": None, "generator": None, "error": None, "adapter": None, "smoke": False, "pipeline": "v3"}
_lock = threading.Lock()  # one GPU, one generation at a time


class ChatRequest(BaseModel):
    message: str
    max_length: int = 700


class ChatResponse(BaseModel):
    response: str


def load(model: str, adapter: str | None, smoke: bool, pipeline: str = "v3"):
    """Load index and model; record the error instead of crashing so /health can report it."""
    import os

    from .generate import Generator
    from .retriever import Retriever
    try:
        if not adapter and not smoke:
            raise RuntimeError("no adapter given: serving requires the fine-tuned adapter (use --smoke for tests)")
        if pipeline == "v4":  # v4 is scored on index_v2 with hybrid retrieval; serve exactly that
            from .bakeoff import INDEX_V2
            STATE["retriever"] = Retriever(INDEX_V2, "hybrid")
        else:
            STATE["retriever"] = Retriever()
        STATE["generator"] = Generator(model, adapter, token=os.environ.get("HF_TOKEN"))
        STATE["adapter"], STATE["smoke"], STATE["pipeline"] = adapter, smoke, pipeline
        log.info("loaded model=%s adapter=%s", model, adapter)
    except Exception as e:  # noqa: BLE001 - any load failure must surface through /health
        STATE["error"] = f"{type(e).__name__}: {e}"
        log.exception("model load failed")


def ready() -> bool:
    return STATE["retriever"] is not None and STATE["generator"] is not None and STATE["error"] is None


@app.get("/health")
def health():
    return {"status": "ok" if ready() else "unavailable", "model_loaded": ready()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not ready():
        raise HTTPException(status_code=503, detail=f"Model not loaded: {STATE['error'] or 'still loading'}")
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty message")
    hits = STATE["retriever"].search(question, TOP_K)
    if STATE["pipeline"] == "v4":
        gen = STATE["generator"]
        with _lock:  # v4 sets its own token budgets per step; max_length doesn't apply
            r = run_batch([question], [hits], lambda convs, mt: [gen(c, max_new_tokens=mt) for c in convs])[0]
        log.info("v4 %s on %r", r["status"], question[:80])
        return ChatResponse(response=r["response"])
    with _lock:
        answer = STATE["generator"](build_messages(question, hits), max_new_tokens=min(req.max_length, MAX_NEW_TOKENS))
    shown, flags = guard(answer, hits)
    if flags:
        log.warning("guard fired on %r: %s", question[:80], flags)
    return ChatResponse(response=shown)


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=["v3", "v4"], default="v3")
    ap.add_argument("--model", default=None, help="default: Llama 3.1 8B for v3, Qwen3-4B-Instruct-2507 for v4")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--smoke", action="store_true", help="allow serving without an adapter (plumbing tests only)")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    model = a.model or ("Qwen/Qwen3-4B-Instruct-2507" if a.pipeline == "v4" else "meta-llama/Llama-3.1-8B-Instruct")
    load(model, a.adapter, a.smoke, a.pipeline)
    uvicorn.run(app, host="0.0.0.0", port=a.port)


if __name__ == "__main__":
    main()
