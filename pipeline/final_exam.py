"""Score the shipped v4 system on the sealed final exam, exactly once (Colab T4).

    python -m pipeline.final_exam --smoke     # plumbing check on 3 gold_v2 questions, never the exam
    python -m pipeline.final_exam             # the sealed exam: 25 questions, scored once

The shipped system is what this runs: hybrid retrieval on index_v2, Qwen3-4B-Instruct-2507 in
4-bit NF4 with the v4 adapter (the precision it was trained in), the v4 pipeline, one question
at a time. Each question is timed end to end (retrieval + every model call), which is the reply
time a user of the Colab demo would see (criterion 7).

Sealing: if the exam report exists, this refuses to run. Answers are appended per question,
so a dropped Colab session resumes the same run instead of starting a second one.
"""
import argparse
import hashlib
import json
import time
from collections import Counter
from statistics import median

from . import tracking
from .bakeoff import INDEX_V2
from .config import DATA, EVAL_DIR, FINAL_EXAM_V1, GOLD_V2, TOP_K
from .eval_answers import summarize
from .grade import grade
from .v4 import run_batch

BASE = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER = DATA / "models" / "v4"
# Final-exam criteria, as pre-registered in reports/V4_PLAN.md (2026-10-17)
CRITERIA = {"wrong_rules_shown": ("==", 0), "must_refuse_refused": ("==", 6),
            "fact_coverage": (">=", 0.60), "over_refusal_rate": ("<=", 0.10)}
SPEED_LIMIT_S = 15  # criterion 7: median reply time


def _passes(value, op, target):
    return value == target if op == "==" else value >= target if op == ">=" else value <= target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 gold_v2 questions; checks the plumbing, not the exam")
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--adapter", default=str(ADAPTER), help='"" for no adapter (local plumbing tests with a tiny model)')
    a = ap.parse_args()

    gold_path, tag = (GOLD_V2, "v4_colab_smoke") if a.smoke else (FINAL_EXAM_V1, "final_exam_v1")
    preds_path, report_path = EVAL_DIR / f"{tag}_predictions.jsonl", EVAL_DIR / f"{tag}_report.json"
    if not a.smoke:
        if report_path.exists():
            raise SystemExit(f"{report_path.name} exists: the final exam is scored once and never re-run")
        expected = (EVAL_DIR / "final_exam_v1.sha256").read_text().split()[0]
        assert hashlib.sha256(gold_path.read_bytes()).hexdigest() == expected, "final_exam_v1.jsonl changed since it was sealed"
    gold = [json.loads(l) for l in gold_path.read_text().splitlines() if l.strip()]
    if a.smoke:
        gold = gold[:3]

    import os

    import torch

    from .generate import Generator
    from .retriever import Retriever
    ret = Retriever(INDEX_V2, "hybrid")
    gen = Generator(a.model, a.adapter or None, token=os.environ.get("HF_TOKEN"))

    def chat(convs, max_tokens):  # one conversation at a time: the same path a single user's question takes
        return [gen(c, max_new_tokens=max_tokens) for c in convs]

    done = {}
    if preds_path.exists():  # resume after a Colab disconnect
        done = {r["id"]: r for r in map(json.loads, preds_path.read_text().splitlines())}
    with preds_path.open("a") as f:
        for g in gold:
            if g["id"] in done:
                continue
            t = time.time()
            hits = ret.search(g["question"], TOP_K)
            r = run_batch([g["question"]], [hits], chat)[0]
            row = {"id": g["id"], "program": g["program"], "must_refuse": g["must_refuse"], **r,
                   "seconds": round(time.time() - t, 2), "grade": grade(r["response"], hits, g),
                   "hit_urls": [h["url"] for h in hits]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[g["id"]] = row
            print(f"{g['id']}: {row['status']} in {row['seconds']} s", flush=True)

    rows = [done[g["id"]] for g in gold]
    overall = summarize(rows)["overall"]
    measured = {"wrong_rules_shown": sum(bool(r["grade"].get("wrong_verdict")) or bool(r["grade"].get("contradicts_source")) for r in rows),
                "must_refuse_refused": sum(r["grade"]["refused"] for r in rows if r["must_refuse"]),
                "fact_coverage": overall["fact_coverage"], "over_refusal_rate": overall["over_refusal_rate"]}
    secs = [r["seconds"] for r in rows]
    report = {"tag": tag, "gold_file": gold_path.name, "model": a.model, "adapter": a.adapter or None,
              "quantization": gen.quantization, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "retriever": "hybrid index_v2", "pipeline": "v4 extract->answer->verify", "grader_version": 3,
              "measured": measured, "status": dict(Counter(r["status"] for r in rows)),
              "median_reply_s": median(secs), "max_reply_s": max(secs), "overall": overall}
    if not a.smoke:
        report["criteria"] = {k: {"target": f"{op} {t}", "value": measured[k], "pass": _passes(measured[k], op, t)}
                              for k, (op, t) in CRITERIA.items()}
        report["criteria"]["median_reply_s"] = {"target": f"<= {SPEED_LIMIT_S}", "value": median(secs),
                                                "pass": median(secs) <= SPEED_LIMIT_S}
        report["passed"] = all(c["pass"] for c in report["criteria"].values())
    report_path.write_text(json.dumps(report, indent=1))
    tracking.record(f"final_exam:{tag}", inputs=[gold_path, ADAPTER / "adapter_model.safetensors", INDEX_V2 / "embeddings.npy"],
                    outputs=[preds_path, report_path], config={k: report[k] for k in ("model", "adapter", "quantization", "gpu", "retriever")},
                    metrics={k: report.get(k) for k in ("measured", "median_reply_s", "max_reply_s", "passed")})
    print(json.dumps({k: report.get(k) for k in ("measured", "median_reply_s", "criteria", "passed")}, indent=1))


if __name__ == "__main__":
    main()
