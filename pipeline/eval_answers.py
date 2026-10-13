"""Answer the gold set with the serving prompt and grade every answer.

    python -m pipeline.eval_answers --tag baseline                       # Llama 3.1 8B, no adapter (Colab GPU)
    python -m pipeline.eval_answers --tag v2 --adapter visa_buddy_data/models/v2
    python -m pipeline.eval_answers --tag smoke --model Qwen/Qwen2.5-0.5B-Instruct --limit 6

Writes eval/<tag>_predictions.jsonl (appended per question, so a dropped Colab session
resumes where it stopped) and eval/<tag>_report.json, then records a ledger run.
A "smoke" run only checks the plumbing; its numbers are never reported as a baseline.
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from . import tracking
from .config import EVAL_DIR, GOLD_V1, PROGRAMS, TOP_K
from .grade import grade
from .prompt import SYSTEM, build_messages
from .retriever import EMB, META, Retriever

METRICS = ["citation_hit", "fact_coverage", "refusal_correct", "unsupported"]


def summarize(rows):
    def rate(rs, key):
        vals = [r["grade"][key] for r in rs if r["grade"][key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def block(rs):
        ans = [r for r in rs if not r["must_refuse"]]
        ref = [r for r in rs if r["must_refuse"]]
        answered = [r for r in ans if not r["grade"]["refused"]]
        return {"metrics_version": 2, "n": len(rs), "n_answerable": len(ans), "n_must_refuse": len(ref),
                # v2 definitions: a refusal is not a citation miss, and refusing when the retrieved
                # sources don't contain the answer is correct (abstention), not a false refusal
                "cite_when_answering": round(sum(r["grade"]["citation_hit"] for r in answered) / len(answered), 3) if answered else None,
                "over_refusal_rate": round(sum(r["grade"]["refused"] and bool(r["grade"].get("answer_in_context")) for r in ans) / len(ans), 3) if ans else None,
                "abstained_answer_not_in_context": sum(r["grade"]["refused"] and r["grade"].get("answer_in_context") is False for r in ans),
                "answer_in_context_rate": round(sum(bool(r["grade"].get("answer_in_context")) for r in ans) / len(ans), 3) if ans else None,
                # v1 definitions, kept for continuity with earlier reports
                "citation_hit": rate(ans, "citation_hit"),  # citations only expected on answers
                "fact_coverage": rate(ans, "fact_coverage"),
                "refusal_correct": rate(rs, "refusal_correct"),
                "refusal_correct_on_must_refuse": rate(ref, "refusal_correct"),
                "false_refusal_rate": round(sum(r["grade"]["refused"] for r in ans) / len(ans), 3) if ans else None,
                "unsupported": rate(ans, "unsupported")}

    by_prog = defaultdict(list)
    for r in rows:
        by_prog[r["program"]].append(r)
    return {"overall": block(rows), "per_program": {p: block(by_prog[p]) for p in PROGRAMS if by_prog[p]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--gold", default=str(GOLD_V1))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    a = ap.parse_args()

    from .generate import Generator
    gold = [json.loads(l) for l in Path(a.gold).read_text().splitlines() if l.strip()]
    if a.limit:
        gold = gold[:: max(1, len(gold) // a.limit)][: a.limit]  # spread across programs
    pred_path = EVAL_DIR / f"{a.tag}_predictions.jsonl"
    done = {}
    if pred_path.exists():
        done = {json.loads(l)["id"]: json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()}
        print(f"resuming: {len(done)} answers already saved")

    ret = Retriever()
    gen = Generator(a.model, a.adapter, token=os.environ.get("HF_TOKEN"))
    with open(pred_path, "a") as f:
        for i, g in enumerate(gold, 1):
            if g["id"] in done:
                continue
            hits = ret.search(g["question"], TOP_K)
            t0 = time.time()
            answer = gen(build_messages(g["question"], hits), a.max_new_tokens)
            row = {"id": g["id"], "program": g["program"], "must_refuse": g["must_refuse"], "question": g["question"],
                   "answer": answer, "retrieved": [{"url": h["url"], "title": h["title"], "score": h["score"]} for h in hits],
                   "grade": grade(answer, hits, g), "latency_s": round(time.time() - t0, 2)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[g["id"]] = row
            gr = row["grade"]
            print(f"[{i}/{len(gold)}] {g['id']} cite={gr['citation_hit']} facts={gr['fact_coverage']} "
                  f"refused={gr['refused']} unsupported={gr['unsupported']} ({row['latency_s']}s)")

    rows = [done[g["id"]] for g in gold if g["id"] in done]
    report = {"tag": a.tag, "model": a.model, "adapter": a.adapter, "quantization": gen.quantization,
              "decoding": "greedy", "top_k": TOP_K, "gold_file": Path(a.gold).name,
              "complete": len(rows) == len(gold), **summarize(rows),
              "median_latency_s": sorted(r["latency_s"] for r in rows)[len(rows) // 2]}
    rep_path = EVAL_DIR / f"{a.tag}_report.json"
    rep_path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["overall"], indent=1))
    tracking.record(f"eval_answers:{a.tag}", inputs=[Path(a.gold), EMB, META], outputs=[pred_path, rep_path],
                    config={"model": a.model, "adapter": a.adapter, "quantization": gen.quantization,
                            "decoding": "greedy", "max_new_tokens": a.max_new_tokens, "top_k": TOP_K,
                            "system_prompt": SYSTEM, "limit": a.limit},
                    metrics={k: report[k] for k in ("overall", "per_program", "complete", "median_latency_s")},
                    notes="smoke test of plumbing only; not a baseline" if a.tag == "smoke" else "")


if __name__ == "__main__":
    main()
