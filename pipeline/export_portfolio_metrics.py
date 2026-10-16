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
from .config import EVAL_DIR, GOLD_V1, REPORTS_DIR

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
}


def _run(run_id):
    if run_id is None:
        return None
    rows = [r for r in tracking.load() if r["run_id"] == run_id]
    if not rows:
        raise SystemExit(f"pinned run {run_id} not in runs/ledger.jsonl")
    return rows[0]


def pct(x):
    return None if x is None else round(100 * x, 1)


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
