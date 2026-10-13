"""Teacher model on Modal: Mistral Small 3.2 24B Instruct (Apache 2.0) served with vLLM.

Same setup that worked for RouteJudge's labeling job (H100, vLLM 0.31.0, Mistral
tokenizer flags, FlashInfer sampler off, weights cached on a volume). Data goes in with
the function call and results come back to this Mac; the job hashes what it received
so the ledger proves which requests were answered.

    caffeinate -i modal run --detach pipeline/teacher_modal.py --phase questions
    caffeinate -i modal run --detach pipeline/teacher_modal.py --phase answers
    caffeinate -i modal run --detach pipeline/teacher_modal.py --phase questions --data v3   # only uncached
    modal run pipeline/teacher_modal.py --phase questions --fetch-only   # if the local client dropped mid-run

The job also saves its result to the `visabuddy-teacher-out` volume, so a dropped local
connection (the first questions run was stopped that way) doesn't lose finished work.

Inputs/outputs live in visa_buddy_data/sft/work/ (built by `python -m pipeline.sft ...`).
"""
import hashlib
import json
from pathlib import Path

import modal

TEACHER = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
GPU = "H100"  # 80 GB: the 24B model in bf16 (~45 GB) plus KV cache fits on one card
VLLM_VERSION = "0.31.0"  # the version RouteJudge's teacher job ran on successfully

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "mistral_common")
    # FlashInfer's sampler JIT-compiles CUDA at startup and needs nvcc, which the slim image lacks.
    # Decoding is greedy, so the faster sampler brings nothing.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("routejudge-hf-cache", create_if_missing=True)  # shared weight cache
out_vol = modal.Volume.from_name("visabuddy-teacher-out", create_if_missing=True)
app = modal.App("visabuddy-teacher", image=image)

DATA = Path(__file__).resolve().parent.parent / "visa_buddy_data"


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@app.function(gpu=GPU, timeout=2 * 60 * 60, volumes={"/root/.cache/huggingface": hf_cache, "/out": out_vol})
def chat(phase: str, convs: list, max_tokens: int) -> dict:
    import mistral_common
    import vllm
    from vllm import LLM, SamplingParams

    # Mistral's own tokenizer/config formats: required for this model under vLLM.
    llm = LLM(model=TEACHER, tokenizer_mode="mistral", config_format="mistral",
              load_format="mistral", max_model_len=8192, enable_prefix_caching=True)
    # temperature 0: the same request always gets the same output, so the data is reproducible
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False)
    res = {"teacher": TEACHER, "vllm_version": vllm.__version__, "mistral_common_version": mistral_common.__version__,
           "gpu": GPU, "received_sha256": _digest(convs), "outputs": [o.outputs[0].text for o in outs]}
    with open(f"/out/{phase}.json", "w") as f:
        json.dump(res, f, ensure_ascii=False)
    out_vol.commit()
    return res


@app.function(volumes={"/out": out_vol})
def fetch(phase: str) -> dict:  # phase may carry a data version, e.g. "v3_answers"
    with open(f"/out/{phase}.json") as f:
        return json.load(f)


@app.local_entrypoint()
def main(phase: str, fetch_only: bool = False, data: str = "v2"):
    # v2 sends every request; later versions send only the uncached ones (`<phase>_todo.jsonl`,
    # written by pipeline.sft) and pipeline.sft merges them with the cache.
    work = DATA / ("sft" if data == "v2" else f"sft_{data}") / "work"
    stem = phase if data == "v2" else f"{phase}_todo"
    req_path = work / f"{stem}_requests.jsonl" if data == "v2" else work / f"{stem}.jsonl"
    out_path = work / f"{stem}_teacher.jsonl"
    vol_key = phase if data == "v2" else f"{data}_{phase}"
    reqs = [json.loads(l) for l in req_path.read_text().splitlines() if l.strip()]
    max_tokens = {"questions": 300, "answers": 500}[phase]
    print(f"sending {len(reqs)} {phase} requests to {GPU}", flush=True)
    res = fetch.remote(vol_key) if fetch_only else chat.remote(vol_key, [r["messages"] for r in reqs], max_tokens)
    assert res["received_sha256"] == _digest([r["messages"] for r in reqs]), "job answered different requests"
    # Each output carries the hash of its request, so pipeline.sft can match it to a request
    # without relying on a request file that a later `select` may rewrite. For data versions
    # after v2, new outputs are appended: earlier teacher runs stay in the cache.
    sha = lambda m: hashlib.sha256(json.dumps(m, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    rows = [{"req_id": r["req_id"], "raw": text, "sha": sha(r["messages"])} for r, text in zip(reqs, res["outputs"])]
    if data != "v2" and out_path.exists():
        seen = {json.loads(l)["sha"] for l in out_path.read_text().splitlines() if l.strip()}
        rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()] + [r for r in rows if r["sha"] not in seen]
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {k: res[k] for k in ("teacher", "vllm_version", "mistral_common_version", "gpu", "received_sha256")}
    (work / f"{stem}_teacher_meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out_path, json.dumps(meta))
