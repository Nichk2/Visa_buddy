"""Run the v4 pipeline for each bake-off model on Modal (vLLM, one L4 per model, in parallel).

    caffeinate -i modal run --detach pipeline/bakeoff_modal.py
    modal run pipeline/bakeoff_modal.py --fetch-only     # if the local client dropped mid-run

Same pattern as the teacher job: inputs go in with the call and are hashed inside the job,
results are also saved to a volume. An L4 (24 GB) is plenty for a 3-4B model in bf16; the
run is small (126 questions x 2 steps per model).
"""
import hashlib
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
VLLM_VERSION = "0.31.0"  # the version the teacher jobs ran on
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})  # same reason as teacher_modal: no nvcc in the slim image
    .add_local_python_source("pipeline")  # the exact v4 pipeline code that will ship
)
hf_cache = modal.Volume.from_name("visabuddy-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("visabuddy-bakeoff-out", create_if_missing=True)
app = modal.App("visabuddy-bakeoff", image=image)


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@app.function(gpu="L4", timeout=60 * 60, volumes={"/root/.cache/huggingface": hf_cache, "/out": out_vol})
def run(model_id: str, items: list) -> dict:
    import vllm
    from vllm import LLM, SamplingParams

    from pipeline.v4 import run_batch

    llm = LLM(model=model_id, max_model_len=8192, gpu_memory_utilization=0.9)
    times = []

    def chat(convs, max_tokens):
        t = time.time()
        outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False)
        times.append((time.time() - t) / max(len(convs), 1))
        return [o.outputs[0].text for o in outs]

    results = run_batch([it["question"] for it in items], [it["hits"] for it in items], chat)
    rows = [{"set": it["set"], "id": it["id"], **r} for it, r in zip(items, results)]
    res = {"model": model_id, "vllm_version": vllm.__version__, "gpu": "L4", "received_sha256": _digest(items),
           "median_latency_s": round(sorted(times)[len(times) // 2], 3) if times else None,
           "note": "latency is per question per step in a batched GPU call; Mac latency is measured separately",
           "rows": rows}
    with open(f"/out/{model_id.replace('/', '__')}.json", "w") as f:
        json.dump(res, f, ensure_ascii=False)
    out_vol.commit()
    return res


@app.function(volumes={"/out": out_vol})
def fetch(model_id: str) -> dict:
    with open(f"/out/{model_id.replace('/', '__')}.json") as f:
        return json.load(f)


@app.local_entrypoint()
def main(fetch_only: bool = False, only: str = ""):
    from pipeline.bakeoff import MODELS, OUT, slug
    items = [json.loads(l) for l in (OUT / "items.jsonl").read_text().splitlines() if l.strip()]
    models = [m for m in MODELS if not only or only in m]
    print(f"{len(items)} questions x {len(models)} models on L4", flush=True)
    calls = [fetch.spawn(m) if fetch_only else run.spawn(m, items) for m in models]  # all models in parallel
    for m, call in zip(models, calls):
        res = call.get()
        assert res["received_sha256"] == _digest(items), f"{m}: job answered different questions"
        (OUT / f"{slug(m)}_raw.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res["rows"]))
        (OUT / f"{slug(m)}_meta.json").write_text(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
        print("wrote", slug(m), flush=True)
