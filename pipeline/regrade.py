"""Re-grade saved answers with a different gold file or grader version. No regeneration.

    python -m pipeline.regrade --tag baseline --gold eval/gold_v2.jsonl --grader 2

Retrieval is re-run (deterministic, same index) to recover the chunk text the model saw,
and it must match the URLs saved with each answer, or the run stops.
"""
import argparse
import json
from pathlib import Path

from . import tracking
from .config import EVAL_DIR, TOP_K
from .eval_answers import summarize
from .grade import grade
from .retriever import EMB, META, Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--grader", type=int, default=2)
    a = ap.parse_args()
    gold = {json.loads(l)["id"]: json.loads(l) for l in Path(a.gold).read_text().splitlines() if l.strip()}
    src = EVAL_DIR / f"{a.tag}_predictions.jsonl"
    preds = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    ret = Retriever()
    rows = []
    for p in preds:
        hits = ret.search(p["question"], TOP_K)
        assert [h["url"] for h in hits] == [x["url"] for x in p["retrieved"]], f"{p['id']}: retrieval differs from the saved run"
        rows.append({**p, "grade": grade(p["answer"], hits, gold[p["id"]], version=a.grader)})
    name = f"{a.tag}__{Path(a.gold).stem}__grader{a.grader}"
    out_pred, out_rep = EVAL_DIR / f"{name}_predictions.jsonl", EVAL_DIR / f"{name}_report.json"
    out_pred.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    base = json.loads((EVAL_DIR / f"{a.tag}_report.json").read_text())
    report = {**{k: base[k] for k in ("tag", "model", "adapter", "quantization", "decoding", "top_k")},
              "gold_file": Path(a.gold).name, "grader_version": a.grader, "regraded_from": src.name, **summarize(rows)}
    out_rep.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["overall"], indent=1))
    tracking.record(f"regrade:{a.tag}", inputs=[src, Path(a.gold), EMB, META], outputs=[out_pred, out_rep],
                    config={"gold": Path(a.gold).name, "grader_version": a.grader},
                    metrics={"overall": report["overall"], "per_program": report["per_program"]})


if __name__ == "__main__":
    main()
