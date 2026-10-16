"""Phase 3 model bake-off: the v4 pipeline with each candidate model, no fine-tuning.

    python -m pipeline.bakeoff prepare              # Mac: hybrid retrieval on index_v2 for every eval question
    modal run pipeline/bakeoff_modal.py             # Modal: run the pipeline per model (vLLM, L4)
    python -m pipeline.bakeoff grade                # Mac: grade every model on every set, ledger rows

Retrieval is done once on the Mac and sent to every model, so the models differ only in what
they do with the same sources. Scores here pick a model; the shipped numbers are re-measured
with the winner in 4-bit MLX on the Mac (reports/V4_PLAN.md).
"""
import json
import sys
from collections import Counter
from pathlib import Path

from . import tracking
from .config import DATA, EVAL_DIR, TOP_K

OUT = EVAL_DIR / "bakeoff"
INDEX_V2 = DATA / "index_v2"
SETS = ["gold_v2", "safety_v1", "rules_v1", "rules_yes_v1"]
MODELS = ["Qwen/Qwen3-4B-Instruct-2507", "microsoft/Phi-4-mini-instruct", "HuggingFaceTB/SmolLM3-3B"]


def slug(model: str) -> str:
    return model.split("/")[-1].lower()


def prepare():
    from .retriever import Retriever
    ret = Retriever(INDEX_V2, "hybrid")
    rows = []
    for s in SETS:
        for g in map(json.loads, (EVAL_DIR / f"{s}.jsonl").read_text().splitlines()):
            hits = ret.search(g["question"], TOP_K)
            rows.append({"set": s, "id": g["id"], "question": g["question"],
                         "hits": [{k: h[k] for k in ("url", "title", "text", "score", "chunk_id")} for h in hits]})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "items.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} questions -> {path}")
    tracking.record("bakeoff_prepare", inputs=[EVAL_DIR / f"{s}.jsonl" for s in SETS] + [INDEX_V2 / "embeddings.npy"],
                    outputs=[path], config={"retriever": "hybrid", "index": "index_v2", "top_k": TOP_K},
                    metrics={"questions": len(rows), "per_set": dict(Counter(r["set"] for r in rows))})


def grade_all():
    from .eval_answers import summarize
    from .grade import grade
    items = {(r["set"], r["id"]): r for r in map(json.loads, (OUT / "items.jsonl").read_text().splitlines())}
    gold = {(s, g["id"]): g for s in SETS for g in map(json.loads, (EVAL_DIR / f"{s}.jsonl").read_text().splitlines())}
    table = {}
    for m in MODELS:
        raw_path = OUT / f"{slug(m)}_raw.jsonl"
        if not raw_path.exists():
            print("missing", raw_path)
            continue
        raw = list(map(json.loads, raw_path.read_text().splitlines()))
        meta = json.loads((OUT / f"{slug(m)}_meta.json").read_text())
        graded = []
        for r in raw:
            key = (r["set"], r["id"])
            g, it = gold[key], items[key]
            graded.append({**r, "program": g["program"], "must_refuse": g["must_refuse"],
                           "grade": grade(r["response"], it["hits"], g)})
        by_set = {}
        for s in SETS:
            rows = [x for x in graded if x["set"] == s]
            by_set[s] = {**summarize(rows)["overall"], "status": dict(Counter(x["status"] for x in rows))}
        wrong_shown = sum(bool(x["grade"].get("wrong_verdict")) or bool(x["grade"].get("contradicts_source")) for x in graded)
        refused_32 = sum(x["grade"]["refused"] for x in graded if x["must_refuse"])
        summary = {"model": m, "wrong_rules_shown": wrong_shown, "must_refuse_refused": refused_32,
                   "median_latency_s_gpu": meta.get("median_latency_s"), "by_set": by_set}
        (OUT / f"{slug(m)}_graded.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in graded))
        (OUT / f"{slug(m)}_report.json").write_text(json.dumps(summary, indent=1))
        tracking.record(f"bakeoff:{slug(m)}", inputs=[raw_path, OUT / "items.jsonl"] + [EVAL_DIR / f"{s}.jsonl" for s in SETS],
                        outputs=[OUT / f"{slug(m)}_report.json", OUT / f"{slug(m)}_graded.jsonl"],
                        config={**meta, "pipeline": "v4 extract->answer->verify", "grader_version": 3},
                        metrics=summary)
        table[m] = summary
    print_table(table)


def print_table(table):
    cols = [("wrong rules shown", lambda s: s["wrong_rules_shown"]),
            ("must-refuse /32", lambda s: s["must_refuse_refused"]),
            ("facts gold", lambda s: s["by_set"]["gold_v2"]["fact_coverage"]),
            ("facts rules", lambda s: s["by_set"]["rules_v1"]["fact_coverage"]),
            ("facts yes", lambda s: s["by_set"]["rules_yes_v1"]["fact_coverage"]),
            ("over-refusal gold", lambda s: s["by_set"]["gold_v2"]["over_refusal_rate"]),
            ("over-refusal rules", lambda s: s["by_set"]["rules_v1"]["over_refusal_rate"])]
    print(f"{'model':30}" + "".join(f"{c[0]:>20}" for c in cols))
    for m, s in table.items():
        print(f"{slug(m):30}" + "".join(f"{(f'{v:.3f}' if isinstance(v, float) else str(v)):>20}" for v in (f(s) for _, f in cols)))


if __name__ == "__main__":
    {"prepare": prepare, "grade": grade_all}[sys.argv[1]]()
