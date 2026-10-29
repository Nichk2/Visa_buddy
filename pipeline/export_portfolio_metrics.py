"""Export the numbers shown in the VisaBuddy portfolio case study.

Same idea as routejudge/experiments/export_portfolio_metrics.py: every figure on the
page traces to a recorded run. Run ids are pinned (not "latest") so a later
experiment can't silently change what the published page claims. Stages that
haven't run yet export as null, and the page should show them as pending.

    python -m pipeline.export_portfolio_metrics                       # -> reports/portfolio_metrics.json
    python -m pipeline.export_portfolio_metrics path/to/metrics.ts     # typed module for fluid-portfolio
"""
import json
import sys
from pathlib import Path

from . import tracking
from .config import DATA, EVAL_DIR, FINAL_EXAM_V1, GOLD_V1, REPORTS_DIR, ROOT

RUNS = {
    "gold": "c1b04c584eef",
    "legacyAudit": "13f38c36e422",
    "crawl": "687fd6d758fd",
    "chunk": "5b1ce79af2cb",
    "index": "2e8249737c2c",
    "retrieval": "933a39ed6a15",
    "baseline": "dae69d130eea",
    "goldV2": "790993006110",
    "retrievalGoldV2": "a6e8c331926c",
    "baselineGoldV2": "f851708cd6b9",
    "trainV3": "90b5b188c81f",
    "evalV3": "651fc2a78bb0",
    "evalSafetyV3": "6ffd3b33d5f9",
    "guardV3": "e4abbf736a8a",
    "guardSafetyV3": "8470aa5d4e82",
    "train": "fc6ace1a5255",
    "adapterEval": "9b184da36903",
    # grader v3 re-scores of baseline and v2, so all three Llama versions share one grader
    "baselineG3": "93078f69d74d",
    "v2G3": "024efc30bf81",
    "safetyBaselineG3": "9d057f83b7a6",
    "safetyV2G3": "28f4d537be17",
    "rulesBaseline": "a99b468f7c60",
    "rulesV2": "3db6af8d30f4",
    "rulesV3": "adc0bec38321",
    # v4: corpus_v2, hybrid retrieval, the quote-first pipeline
    "crawlV2": "bf77968d58de",
    "chunkV2": "ddaf0f49c39c",
    "retrievalDenseV2": "e3744a44bc34",
    "retrievalHybridV2": "a15854d10e8b",
    "teacherGateMistral": "fe5343b8e4f2",
    "teacherGateQwen30b": "e146c3dfa6c5",
    "sftV4": "944f0d28b0fb",
    "trainV4": "e7e2ae821043",
    "compareUntuned": "3f58c87dcbf9",
    "compareV4": "5caba606ce97",
    "examFreeze": "1086a5ffbd1c",
    "finalExam": "05d649923b28",
    "costs": "0b71794a82bb",
}
# Modal's listed T4 rate (modal billing rates, 2026-10-29), for the cost-per-1,000-answers estimate only
T4_USD_PER_HOUR = 0.59


def _run(run_id):
    if run_id is None:
        return None
    rows = [r for r in tracking.load() if r["run_id"] == run_id]
    if not rows:
        raise SystemExit(f"pinned run {run_id} not in runs/ledger.jsonl")
    return rows[0]


def pct(x):
    return None if x is None else round(100 * x, 1)


def _pinned_file(run, rel):
    """Read a run's output file, refusing if it changed since the run recorded it."""
    path = ROOT / rel
    assert tracking.sha256(path) == run["outputs"][rel], f"{rel} changed since run {run['run_id']}"
    return path


def _versions(r):
    """Same measures for every version. baseline-v3: Llama 3.1 8B, dense retrieval, one-step prompt.
    v4: Qwen3-4B, hybrid retrieval on corpus_v2, quote-first pipeline (system-to-system comparison)."""
    def llama(gold, safety, rules):
        g, s, ru = gold["metrics"]["overall"], safety["metrics"]["overall"], rules["metrics"]["overall"]
        return {"factCoverageGold": pct(g["fact_coverage"]), "factCoverageRules": pct(ru["fact_coverage"]),
                "overRefusalGold": pct(g["over_refusal_rate"]),
                "mustRefuseRefused": round(g["refusal_correct_on_must_refuse"] * g["n_must_refuse"]
                                           + s["refusal_correct_on_must_refuse"] * s["n_must_refuse"])}

    def qwen(run):
        b = run["metrics"]["by_set"]
        return {"factCoverageGold": pct(b["gold_v2"]["fact_coverage"]), "factCoverageRules": pct(b["rules_v1"]["fact_coverage"]),
                "overRefusalGold": pct(b["gold_v2"]["over_refusal_rate"]), "mustRefuseRefused": run["metrics"]["must_refuse_refused"]}

    return {"baseline": llama(r["baselineG3"], r["safetyBaselineG3"], r["rulesBaseline"]),
            "v2": llama(r["v2G3"], r["safetyV2G3"], r["rulesV2"]),
            "v3": llama(r["evalV3"], r["evalSafetyV3"], r["rulesV3"]),
            "v4": qwen(r["compareV4"]), "mustRefuseTotal": 32}


def _compare(run):
    m, b = run["metrics"], run["metrics"]["by_set"]
    return {"wrongRulesShown": m["wrong_rules_shown"], "mustRefuseRefused": m["must_refuse_refused"],
            "factCoverage": {s: pct(b[s]["fact_coverage"]) for s in ("gold_v2", "rules_v1", "rules_yes_v1")},
            "overRefusal": {s: pct(b[s]["over_refusal_rate"]) for s in ("gold_v2", "rules_v1")},
            "wrongVerdictYes": round(b["rules_yes_v1"]["wrong_verdict_rate"] * b["rules_yes_v1"]["n_answerable"])}


def _exam(run):
    preds = [json.loads(l) for l in _pinned_file(run, "eval/final_exam_v1_predictions.jsonl").read_text().splitlines()]
    report = json.loads(_pinned_file(run, "eval/final_exam_v1_report.json").read_text())
    gold = {g["id"]: g for g in map(json.loads, FINAL_EXAM_V1.read_text().splitlines())}
    wrong = [p for p in preds if p["grade"].get("wrong_verdict") or p["grade"].get("contradicts_source")]
    return {"questions": len(preds), "answerable": report["overall"]["n_answerable"], "mustRefuse": report["overall"]["n_must_refuse"],
            "gpu": report["gpu"], "quantization": report["quantization"], "passed": report["passed"],
            "criteria": {k: {"target": c["target"], "value": c["value"], "pass": c["pass"]} for k, c in report["criteria"].items()},
            "medianReplyS": report["median_reply_s"], "maxReplyS": report["max_reply_s"],
            "replySeconds": [{"id": p["id"], "s": p["seconds"], "mustRefuse": p["must_refuse"]} for p in preds],
            "wrongAnswers": [{"id": p["id"], "question": p["question"], "expectedVerdict": gold[p["id"]].get("verdict"),
                              "expectedFact": gold[p["id"]]["expected_facts"][0], "answer": p["answer_raw"].strip(),
                              "quotesShown": len(p["quotes"])} for p in wrong]}


def _loss_curves(r):
    """v1's chart was a formula, 2.5*exp(-0.01*step) + 0.1*random(), over 500 steps; its smooth part is
    drawn here (the noise term can't be reproduced). Next to it, the real v4 training loss."""
    import math
    state = json.loads((DATA / "models" / "v4" / "trainer_state.json").read_text())
    real = [{"step": h["step"], "loss": round(h["loss"], 4)} for h in state["log_history"] if "loss" in h]
    steps = r["legacyAudit"]["metrics"]["v1_plotted_steps"]
    formula = [{"step": s, "loss": round(2.5 * math.exp(-0.01 * s), 4)} for s in range(0, steps + 1, 5)]
    return {"v1Formula": formula, "v1FormulaText": r["legacyAudit"]["metrics"]["v1_loss_formula"],
            "v4Real": real, "v4Steps": state["global_step"],
            "v4ValidByEpoch": r["trainV4"]["metrics"]["valid_by_epoch"], "v4SelectedEpoch": r["trainV4"]["metrics"]["selected_epoch"]}


def build():
    r = {k: _run(v) for k, v in RUNS.items()}
    audit = r["legacyAudit"]["metrics"]
    retr = r["retrieval"]["metrics"]
    legacy_rows = json.loads((EVAL_DIR / "legacy_audit.json").read_text())["old_queries"]
    return {
        "provenance": {"evalSet": "eval/gold_v1.jsonl", "evalSha256": tracking.sha256(GOLD_V1),
                       "runs": RUNS},
        "headline": {
            "claimedScore": audit["v1_claimed_overall_score"],
            "claimedTestQueries": audit["v1_claimed_test_queries"],
            "claimedRetrievalSuccess": pct(audit["v1_marked_successful"] / audit["old_queries"]),
            "measuredRecallAt5": pct(retr["answerable"]["recall@5"]),
        },
        "v1": {
            "pagesCrawled": audit["v1_pages_crawled"],
            "pagesByProgram": audit["v1_pages_by_program"],
            "sftRows": audit["v1_sft_rows"],
            "sftRowsSimilarityGe05": audit["v1_sft_rows_similarity_ge_0_5"],
            "lossCurveFormula": audit["v1_loss_formula"],
            "lossCurvePlottedSteps": audit["v1_plotted_steps"],
            "oldQueriesTop1RightProgram": audit["v1_top1_right_program"],
            "oldQueriesCompared": audit["compared"],
        },
        "v2": {
            "goldQuestions": r["gold"]["metrics"]["rows"],
            "goldAnswerable": r["gold"]["metrics"]["answerable"],
            "goldMustRefuse": r["gold"]["metrics"]["must_refuse"],
            "pagesCrawled": r["crawl"]["metrics"]["docs"],
            "pagesByProgram": r["crawl"]["metrics"]["docs_per_program"],
            "chunks": r["chunk"]["metrics"]["chunks"],
            "pagesHeldOut": r["chunk"]["metrics"]["pages_per_split"]["gold_heldout"],
            "oldQueriesTop1RightProgram": audit["v2_top1_right_program"],
            "retrieval": {
                "overall": {k: pct(retr["answerable"][k]) for k in ("recall@1", "recall@3", "recall@5", "recall@10", "fact_coverage@5")},
                "perProgram": {p: {"n": v["n"], "recallAt5": pct(v["recall@5"]), "factCoverageAt5": pct(v["fact_coverage@5"])}
                               for p, v in retr["per_program"].items()},
            },
        },
        "oldQueries": [{k: q[k] for k in ("id", "query", "want", "v1_top_preview", "v1_top_program", "v2_top_title", "v2_top_program")}
                       for q in legacy_rows],
        "baseline": r["baseline"] and r["baseline"]["metrics"],
        "measurementAudit": {
            "goldV2Sha256": r["goldV2"]["outputs"]["eval/gold_v2.jsonl"],
            "extraSourceUrls": r["goldV2"]["metrics"]["extra_source_urls"],
            "recallAt5": {"goldV1": pct(retr["answerable"]["recall@5"]),
                          "goldV2": pct(r["retrievalGoldV2"]["metrics"]["answerable"]["recall@5"])},
            "baselineFactCoverage": {"goldV1": pct(r["baseline"]["metrics"]["overall"]["fact_coverage"]),
                                     "goldV2": pct(r["baselineGoldV2"]["metrics"]["overall"]["fact_coverage"])},
        },
        "baselineGoldV2": r["baselineGoldV2"]["metrics"],
        "v3": {"training": r["trainV3"]["metrics"], "gold": r["evalV3"]["metrics"]["overall"],
               "safety": r["evalSafetyV3"]["metrics"]["overall"], "guardGold": r["guardV3"]["metrics"],
               "guardSafety": r["guardSafetyV3"]["metrics"]},
        "training": r["train"] and r["train"]["metrics"],
        "adapter": r["adapterEval"] and r["adapterEval"]["metrics"],
        "v4": {
            "pagesCrawled": r["crawlV2"]["metrics"]["docs"],
            "pagesByProgram": r["crawlV2"]["metrics"]["docs_per_program"],
            "chunks": r["chunkV2"]["metrics"]["chunks"],
            "retrieval": {
                "recallAt5": {m: pct(r[k]["metrics"]["answerable"]["recall@5"]) for m, k in (("dense", "retrievalDenseV2"), ("hybrid", "retrievalHybridV2"))},
                "perProgram": {p: {"n": v["n"], "dense": pct(r["retrievalDenseV2"]["metrics"]["per_program"][p]["recall@5"]),
                                   "hybrid": pct(v["recall@5"])}
                               for p, v in r["retrievalHybridV2"]["metrics"]["per_program"].items()},
                "citizenshipTarget": 70.0,
            },
            "teacherGates": {k: {"wrongRulesShown": r[k]["metrics"]["wrong_rules_shown"], "mustRefuseRefused": r[k]["metrics"]["must_refuse_refused"],
                                 "thresholds": r[k]["metrics"]["thresholds"]} for k in ("teacherGateMistral", "teacherGateQwen30b")},
            "sftRows": {"train": r["sftV4"]["metrics"]["train"], "valid": r["sftV4"]["metrics"]["valid"]},
            "lossCurves": _loss_curves(r),
            "compare": {"untuned": _compare(r["compareUntuned"]), "fineTuned": _compare(r["compareV4"])},
        },
        "versions": _versions(r),
        "finalExam": _exam(r["finalExam"]),
        "costs": {**{k: r["costs"]["metrics"][k] for k in ("modalTotal", "gpuOnly", "byRole", "byGpu", "period", "paidWithCredits")},
                  "colabUsd": 0, "t4UsdPerHour": T4_USD_PER_HOUR,
                  # estimate, not a measurement: the exam's median reply time billed at the T4 rate
                  "estUsdPer1000AnswersT4": round(1000 * _exam(r["finalExam"])["medianReplyS"] / 3600 * T4_USD_PER_HOUR, 2)},
    }


def main(out: Path):
    data = build()
    body = json.dumps(data, indent=2, ensure_ascii=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".ts":
        out.write_text("// GENERATED by Visa_buddy/pipeline/export_portfolio_metrics.py. Do not edit by hand.\n"
                       "// Every number comes from the VisaBuddy run ledger; provenance lists the exact runs.\n\n"
                       f"export const visabuddyMetrics = {body} as const;\n")
    else:
        out.write_text(body + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPORTS_DIR / "portfolio_metrics.json")
