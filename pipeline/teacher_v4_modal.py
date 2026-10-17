"""Teacher (Mistral Small 3.2 24B) running the v4 pipeline on Modal, for phase 5 data and its gate.

    caffeinate -i modal run --detach pipeline/teacher_v4_modal.py
    modal run pipeline/teacher_v4_modal.py --fetch-only

One job, two batches: (1) the 126 eval questions from the bake-off, only to gate the teacher
(never used for training); (2) the training items from `pipeline.sft_v4 items`. Both run the
exact pipeline.v4.run_batch the app ships, so training targets are verified quotes and checked
answers. Same vLLM setup as teacher_modal.py.
"""
import hashlib
import json
from pathlib import Path

import modal

TEACHER = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
ROOT = Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.31.0", "mistral_common")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("pipeline")
)
hf_cache = modal.Volume.from_name("routejudge-hf-cache", create_if_missing=True)  # Mistral weights cached here
out_vol = modal.Volume.from_name("visabuddy-teacher-out", create_if_missing=True)
app = modal.App("visabuddy-teacher-v4", image=image)


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@app.function(gpu="H100", timeout=2 * 60 * 60, volumes={"/root/.cache/huggingface": hf_cache, "/out": out_vol})
def run(batches: dict) -> dict:
    import vllm
    from vllm import LLM, SamplingParams

    from pipeline.v4 import run_batch

    llm = LLM(model=TEACHER, tokenizer_mode="mistral", config_format="mistral", load_format="mistral",
              max_model_len=8192, enable_prefix_caching=True)

    def chat(convs, max_tokens):
        outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=max_tokens), use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    res = {"teacher": TEACHER, "vllm_version": vllm.__version__, "received_sha256": _digest(batches), "batches": {}}
    for name, items in batches.items():
        results = run_batch([it["question"] for it in items], [it["hits"] for it in items], chat)
        res["batches"][name] = [{**{k: it[k] for k in ("set", "id", "item_id") if k in it}, **r} for it, r in zip(items, results)]
    with open("/out/v4_pipeline.json", "w") as f:
        json.dump(res, f, ensure_ascii=False)
    out_vol.commit()
    return res


@app.function(volumes={"/out": out_vol})
def fetch() -> dict:
    with open("/out/v4_pipeline.json") as f:
        return json.load(f)


@app.local_entrypoint()
def main(fetch_only: bool = False):
    work = ROOT / "visa_buddy_data" / "sft_v4" / "work"
    load = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    batches = {"gate": load(ROOT / "eval" / "bakeoff" / "items.jsonl"), "train": load(work / "pipeline_items.jsonl")}
    print({k: len(v) for k, v in batches.items()}, flush=True)
    res = fetch.remote() if fetch_only else run.remote(batches)
    assert res["received_sha256"] == _digest(batches), "job processed different items"
    for name, fname in (("gate", "teacher_gate_raw.jsonl"), ("train", "teacher_pipeline_raw.jsonl")):
        (work / fname).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res["batches"][name]))
    (work / "teacher_v4_meta.json").write_text(json.dumps({k: res[k] for k in ("teacher", "vllm_version", "received_sha256")}, indent=1))
    print("wrote", work, flush=True)
