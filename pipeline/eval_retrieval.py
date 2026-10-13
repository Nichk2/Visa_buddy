"""Retrieval check on the frozen gold set.

    python -m pipeline.eval_retrieval [eval/gold_v1.jsonl]

Answerable question: hit@k if any of its source_urls is among the top-k chunk URLs.
must_refuse question: success if no top-5 chunk matches one of the row's answer_markers
(regexes for text that would actually answer it). Rows with no markers (personal
assessment, legal strategy) can't be answered by any page, so retrieval passes them
trivially; they are counted separately so they don't inflate the headline number.
Word overlap is printed for debugging only and never decides success.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from . import tracking
from .config import EVAL_DIR, GOLD_V1, PROGRAMS, TOP_K
from .retriever import EMB, META, Retriever
from .text import fact_in, tokens

KS = [1, 3, 5, 10]
PRED = EVAL_DIR / "retrieval_predictions.jsonl"
REPORT = EVAL_DIR / "retrieval_report.json"
TARGET = 0.7


def main(gold_path: Path = GOLD_V1):
    suffix = "" if Path(gold_path).name == GOLD_V1.name else "_" + Path(gold_path).stem  # e.g. _gold_v2
    PRED = EVAL_DIR / f"retrieval_predictions{suffix}.jsonl"
    REPORT = EVAL_DIR / f"retrieval_report{suffix}.json"
    gold = [json.loads(l) for l in Path(gold_path).read_text().splitlines() if l.strip()]
    ret = Retriever()
    preds = []
    for g in gold:
        hits = ret.search(g["question"], k=max(KS))
        urls = [h["url"] for h in hits]
        row = {"id": g["id"], "program": g["program"], "must_refuse": g["must_refuse"],
               "question": g["question"], "source_urls": g["source_urls"],
               "top": [{k: h[k] for k in ("rank", "score", "url", "title", "section", "program", "chunk_id")} for h in hits],
               "top1_score": hits[0]["score"]}
        if not g["must_refuse"]:
            first = next((i + 1 for i, u in enumerate(urls) if u in g["source_urls"]), None)
            row["first_relevant_rank"] = first
            for k in KS:
                row[f"hit@{k}"] = first is not None and first <= k
            row["top1_program_match"] = hits[0]["program"] == g["program"]
            ctx = "\n".join(h["text"] for h in hits[:TOP_K])
            sup = [f for f in g["expected_facts"] if fact_in(f, ctx)]
            row["facts_in_top5"] = len(sup)
            row["facts_total"] = len(g["expected_facts"])
        else:
            markers = g.get("answer_markers", [])
            matched = [m for m in markers for h in hits[:TOP_K] if re.search(m, h["text"], re.I)]
            row["refusal_reason"] = g.get("refusal_reason")
            row["has_markers"] = bool(markers)
            row["answer_like_chunk_in_top5"] = bool(matched)
            row["retrieval_ok"] = not matched
        q = set(tokens(g["question"]))  # debug only
        row["debug_word_overlap_top1"] = round(len(q & set(tokens(hits[0]["text"]))) / max(len(q), 1), 3)
        preds.append(row)

    with open(PRED, "w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    ans = [p for p in preds if not p["must_refuse"]]
    ref = [p for p in preds if p["must_refuse"]]

    def rates(rows):
        n = len(rows)
        out = {"n": n}
        for k in KS:
            out[f"recall@{k}"] = round(sum(r[f"hit@{k}"] for r in rows) / n, 3) if n else None
        out["mrr@10"] = round(sum(1 / r["first_relevant_rank"] for r in rows if r["first_relevant_rank"]) / n, 3) if n else None
        out["fact_coverage@5"] = round(sum(r["facts_in_top5"] / r["facts_total"] for r in rows) / n, 3) if n else None
        out["all_facts@5"] = round(sum(r["facts_in_top5"] == r["facts_total"] for r in rows) / n, 3) if n else None
        out["top1_program_match"] = round(sum(r["top1_program_match"] for r in rows) / n, 3) if n else None
        return out

    per_program = {p: rates([r for r in ans if r["program"] == p]) for p in PROGRAMS if p != "out_of_scope"}
    with_markers = [r for r in ref if r["has_markers"]]
    report = {
        "gold_file": str(Path(gold_path).name),
        "top_k": TOP_K,
        "answerable": rates(ans),
        "per_program": per_program,
        "must_refuse": {
            "n": len(ref),
            "n_with_answer_markers": len(with_markers),
            "retrieval_ok_with_markers": round(sum(r["retrieval_ok"] for r in with_markers) / len(with_markers), 3) if with_markers else None,
            "no_marker_rows_pass_trivially": len(ref) - len(with_markers),
            "by_reason": {k: {"n": len(v), "ok": sum(r["retrieval_ok"] for r in v)} for k, v in group(ref, "refusal_reason").items()},
        },
        "top1_score": {
            "answerable_mean": round(sum(r["top1_score"] for r in ans) / len(ans), 3),
            "must_refuse_mean": round(sum(r["top1_score"] for r in ref) / len(ref), 3) if ref else None,
        },
        "target_recall@5": TARGET,
        "passes_target": rates(ans)["recall@5"] >= TARGET,
        "misses_at_5": [{"id": r["id"], "question": r["question"], "expected": r["source_urls"],
                         "got_top3": [t["url"] for t in r["top"][:3]]} for r in ans if not r["hit@5"]],
    }
    REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print_table(report)
    tracking.record("eval_retrieval", inputs=[gold_path, EMB, META], outputs=[PRED, REPORT],
                    config={"top_k": TOP_K, "ks": KS, "target": TARGET},
                    metrics={k: report[k] for k in ("answerable", "per_program", "must_refuse", "top1_score", "passes_target")})
    return report


def group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return out


def print_table(rep):
    print(f"\n{'program':<14} {'n':>3} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'facts@5':>8}")
    for name, r in list(rep["per_program"].items()) + [("ALL answerable", rep["answerable"])]:
        print(f"{name:<14} {r['n']:>3} {r['recall@1']:>6.2f} {r['recall@3']:>6.2f} {r['recall@5']:>6.2f} {r['recall@10']:>6.2f} {r['mrr@10']:>6.2f} {r['fact_coverage@5']:>8.2f}")
    m = rep["must_refuse"]
    print(f"\nmust_refuse: {m['n']} rows; {m['n_with_answer_markers']} with answer markers, "
          f"retrieval_ok={m['retrieval_ok_with_markers']}; {m['no_marker_rows_pass_trivially']} pass trivially")
    print(f"recall@5 target {rep['target_recall@5']}: {'PASS' if rep['passes_target'] else 'FAIL'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else GOLD_V1)
