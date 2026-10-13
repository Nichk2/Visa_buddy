"""Audit of the v1 system, recorded as a ledger run so the case study can cite it.

    python -m pipeline.legacy_audit

Reads only files already in the repo (the v1 notebook outputs, the v1 metrics folder)
and, for a like-for-like check, runs the v2 retriever on v1's own 10 test queries.
"""
import csv
import json
import re
from collections import Counter

from . import tracking
from .config import EVAL_DIR, ROOT
from .fetch import program_for

NB = ROOT / "src" / "notebooks" / "Visa Buddy jupyter with outputs and measures.ipynb"
OLD = ROOT / "visa_buddy_data" / "metrics"
OLD_CSV = OLD / "rag_evaluation_results_direct.csv"
OLD_REPORT = OLD / "evaluation_report_20251125_150653.json"
OUT = EVAL_DIR / "legacy_audit.json"

# Program each v1 test query asks about (hand label). Q5 ("How long does visa processing
# take?") isn't about one program and is left out of the program-match comparison.
QUERY_PROGRAM = {"Q1": "express_entry", "Q2": "study_permit", "Q3": "work_permit", "Q4": "pgwp",
                 "Q6": "family", "Q7": "study_permit", "Q8": "citizenship", "Q9": "study_permit",
                 "Q10": "visitor"}
# Program of v1's top chunk, labelled by reading top_doc_preview in the v1 CSV.
# All four distinct previews are Express Entry pages (profile, CRS score, PR application, COPR).
V1_TOP_PROGRAM = {
    "Create your profile and enter the pool": "express_entry",
    "recalculated score is less than the lowest score": "express_entry",
    "Apply for permanent residence": "express_entry",
    "Check your score": "express_entry",
    "If we approve your application": "express_entry",
}


def _outputs(cell):
    return "".join("".join(o.get("text", "")) for o in cell.get("outputs", []))


def notebook_facts():
    cells = json.loads(NB.read_text())["cells"]
    saved, sims, sim_fn = [], [], False
    for c in cells:
        out, src = _outputs(c), "".join(c["source"])
        saved += [u.split("#")[0] for u in re.findall(r"\[Depth \d\] Saved (\S+)", out)]
        sims += [float(x) for x in re.findall(r"Created example \d+: .*?\(similarity: ([0-9.]+)\)", out)]
        sim_fn |= "def simulate_training_metrics" in src and "np.exp(-0.01" in src
    pages = sorted(set(saved))
    by_prog = Counter(program_for(u) or "landing/other" for u in pages)
    return {
        "v1_pages_crawled": len(pages),
        "v1_pages_by_program": dict(by_prog),
        "v1_sft_rows": len(sims),
        "v1_sft_rows_similarity_ge_0_5": sum(s >= 0.5 for s in sims),
        "v1_sft_min_similarity": min(sims) if sims else None,
        "v1_loss_curve_is_formula": sim_fn,
        "v1_loss_formula": "2.5*exp(-0.01*step) + 0.1*random()",
        "v1_plotted_steps": 500,
    }


def old_queries():
    from .retriever import Retriever
    ret = Retriever()
    rows = []
    for r in csv.DictReader(open(OLD_CSV)):
        qid = r["query_id"]
        preview = r["top_doc_preview"].split("\n")[0]
        v1_prog = next((p for k, p in V1_TOP_PROGRAM.items() if preview.startswith(k) or k in preview), None)
        hit = ret.search(r["query"], 1)[0]
        rows.append({"id": qid, "query": r["query"], "want": QUERY_PROGRAM.get(qid),
                     "v1_marked_successful": r["successful_retrieval"] == "True",
                     "v1_top_preview": preview[:80], "v1_top_program": v1_prog,
                     "v2_top_title": hit["title"], "v2_top_url": hit["url"], "v2_top_program": hit["program"]})
    scored = [r for r in rows if r["want"]]
    return rows, {
        "old_queries": len(rows),
        "v1_marked_successful": sum(r["v1_marked_successful"] for r in rows),
        "compared": len(scored),
        "v1_top1_right_program": sum(r["v1_top_program"] == r["want"] for r in scored),
        "v2_top1_right_program": sum(r["v2_top_program"] == r["want"] for r in scored),
    }


def main():
    old = json.loads(OLD_REPORT.read_text())
    facts = notebook_facts()
    facts.update({"v1_claimed_overall_score": old["performance_rating"]["overall_system_score"],
                  "v1_claimed_test_queries": old["rag_performance"]["total_test_queries"],
                  "v1_training_progress_field": old["training_metrics"]["training_progress"]})
    rows, cmp = old_queries()
    facts.update(cmp)
    OUT.write_text(json.dumps({"facts": facts, "old_queries": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(facts, indent=1))
    tracking.record("legacy_audit", inputs=[NB, OLD_CSV, OLD_REPORT], outputs=[OUT],
                    config={"query_program_labels": QUERY_PROGRAM, "v1_top_program_labels": V1_TOP_PROGRAM},
                    metrics=facts)


if __name__ == "__main__":
    main()
