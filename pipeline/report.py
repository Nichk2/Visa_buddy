"""Build portfolio charts and reports/PROGRESS.md from the run ledger and eval files.

    python -m pipeline.report

Nothing here computes a model metric. It only plots numbers that a pipeline stage
already wrote to runs/ledger.jsonl or eval/*.json(l), and every figure caption
names the ledger run_id it came from.
"""
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import tracking
from .config import EVAL_DIR, PROGRAMS, REPORTS_DIR, ROOT

FIG = REPORTS_DIR / "figures"
# Reference palette (dataviz skill): slot 1 blue, slot 2 orange; text tokens for ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"express_entry": "Express Entry", "study_permit": "Study permit", "work_permit": "Work permit",
          "pgwp": "PGWP", "visitor": "Visitor", "family": "Family", "citizenship": "Citizenship"}


def _style(ax, title, subtitle=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=18 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2)


def _save(fig, name, source):
    fig.text(0.01, 0.01, f"source: {source}", fontsize=7, color=INK2)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)
    return f"figures/{name}"


def fig_recall_by_program(rep, run_id):
    progs = [p for p in PROGRAMS if p in rep["per_program"]]
    vals = [rep["per_program"][p]["recall@5"] for p in progs]
    ns = [rep["per_program"][p]["n"] for p in progs]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = range(len(progs))[::-1]
    ax.barh(list(y), vals, color=BLUE, height=0.55)
    ax.axvline(rep["target_recall@5"], color=INK2, linestyle="--", linewidth=1)
    ax.text(rep["target_recall@5"], -0.75, f" target {rep['target_recall@5']}", color=INK2, fontsize=8, va="top")
    for yi, v, n in zip(y, vals, ns):
        ax.text(v + 0.01, yi, f"{v:.2f}  (n={n})", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y), [LABELS[p] for p in progs])
    ax.set_xlim(0, 1.18)
    ax.set_ylim(-1.0, len(progs) - 0.5)
    a = rep["answerable"]
    _style(ax, "Retrieval recall@5 by program",
           f"share of gold questions whose official source page is in the top 5 chunks (overall {a['recall@5']:.2f}, n={a['n']})")
    return _save(fig, "retrieval_recall5_by_program.png", f"eval/retrieval_report.json, ledger run {run_id}")


def fig_recall_at_k(rep, run_id):
    ks = [1, 3, 5, 10]
    vals = [rep["answerable"][f"recall@{k}"] for k in ks]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ks, vals, color=BLUE, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    for k, v in zip(ks, vals):
        ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK)
    ax.set_xticks(ks, [f"k={k}" for k in ks])
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    _style(ax, "Recall@k, all answerable questions")
    return _save(fig, "retrieval_recall_at_k.png", f"eval/retrieval_report.json, ledger run {run_id}")


def fig_top1_scores(preds, run_id):
    ans = [p["top1_score"] for p in preds if not p["must_refuse"]]
    ref = [p["top1_score"] for p in preds if p["must_refuse"]]
    fig, ax = plt.subplots(figsize=(7, 2.6))
    import random
    rnd = random.Random(0)  # jitter only, for legibility
    ax.scatter(ans, [1 + rnd.uniform(-0.12, 0.12) for _ in ans], s=28, color=BLUE, edgecolor=SURFACE, linewidth=1.5, label="answerable")
    ax.scatter(ref, [0 + rnd.uniform(-0.12, 0.12) for _ in ref], s=28, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, label="must refuse")
    ax.set_yticks([0, 1], [f"must refuse (n={len(ref)})", f"answerable (n={len(ans)})"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("cosine similarity of the top-1 chunk", fontsize=9, color=INK2)
    _style(ax, "Top-1 similarity: answerable vs must-refuse questions",
           "overlapping ranges mean a score threshold can't decide refusals; the generator has to")
    return _save(fig, "retrieval_top1_scores.png", f"eval/retrieval_predictions.jsonl, ledger run {run_id}")


def fig_corpus(chunk_run):
    m = chunk_run["metrics"]
    progs = [p for p in PROGRAMS if p in m["chunks_per_program"]]
    vals = [m["chunks_per_program"][p] for p in progs]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = range(len(progs))[::-1]
    ax.barh(list(y), vals, color=BLUE, height=0.55)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, str(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y), [LABELS[p] for p in progs])
    _style(ax, "Corpus: chunks per program",
           f"{m['docs']} official pages -> {m['chunks']} chunks; {m['pages_per_split']['gold_heldout']} pages held out from training")
    return _save(fig, "corpus_chunks_by_program.png", f"ledger run {chunk_run['run_id']} (stage chunk)")


def fig_coverage(audit_run, crawl_run):
    v1 = audit_run["metrics"]["v1_pages_by_program"]
    v2 = crawl_run["metrics"]["docs_per_program"]
    progs = [p for p in PROGRAMS if p in LABELS]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = list(range(len(progs)))[::-1]
    h = 0.36
    ax.barh([v + h / 2 for v in y], [v1.get(p, 0) for p in progs], height=h, color=ORANGE, label="v1 crawl")
    ax.barh([v - h / 2 for v in y], [v2.get(p, 0) for p in progs], height=h, color=BLUE, label="v2 crawl")
    for yi, p in zip(y, progs):
        ax.text(v1.get(p, 0) + 1.5, yi + h / 2, str(v1.get(p, 0)), va="center", fontsize=8, color=INK)
        ax.text(v2.get(p, 0) + 1.5, yi - h / 2, str(v2.get(p, 0)), va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [LABELS[p] for p in progs])
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax, "Official pages indexed per program",
           f"v1: {audit_run['metrics']['v1_pages_crawled']} pages, all Express Entry or the landing page. "
           f"v2: {crawl_run['metrics']['docs']} pages across 7 programs")
    return _save(fig, "v1_v2_coverage.png", f"ledger runs {audit_run['run_id']} (legacy_audit), {crawl_run['run_id']} (crawl)")


def fig_old_queries(audit_run):
    m = audit_run["metrics"]
    n = m["compared"]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    rows = [("v1 reported \"successful\"", m["v1_marked_successful"] / m["old_queries"], ORANGE,
             f"{m['v1_marked_successful']}/{m['old_queries']} (any chunk = success)"),
            ("v1 top result in the right program", m["v1_top1_right_program"] / n, ORANGE, f"{m['v1_top1_right_program']}/{n}"),
            ("v2 top result in the right program", m["v2_top1_right_program"] / n, BLUE, f"{m['v2_top1_right_program']}/{n}")]
    y = list(range(len(rows)))[::-1]
    for yi, (_, v, c, lab) in zip(y, rows):
        ax.barh(yi, v, color=c, height=0.55)
        ax.text(v + 0.01, yi, lab, va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, 1.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    _style(ax, "v1's own 10 test queries, re-checked", "same questions; v1 counted every query a success")
    return _save(fig, "v1_v2_old_queries.png", f"eval/legacy_audit.json, ledger run {audit_run['run_id']}")


def legacy_audit():
    """Facts about the old run, each read from a file in the repo."""
    old = ROOT / "visa_buddy_data" / "metrics"
    r1 = json.loads((old / "evaluation_report_20251125_150653.json").read_text())
    rows = [l.split(",") for l in (old / "rag_evaluation_results_direct.csv").read_text().splitlines()[1:] if l.startswith("Q")]
    return {
        "old_overall_score": r1["performance_rating"]["overall_system_score"],
        "old_test_queries": r1["rag_performance"]["total_test_queries"],
        "old_training_progress": r1["training_metrics"]["training_progress"],
        "old_csv_queries": len(rows),
    }


def main():
    lines = ["# Visa Buddy v2: progress log", "",
             "Generated by `python -m pipeline.report` from `runs/ledger.jsonl` and `eval/`. "
             "Every number below links to the ledger run that produced it.", ""]
    gold = tracking.load("gold_freeze")
    crawl = tracking.load("crawl")
    chunk = tracking.load("chunk")
    retr = tracking.load("eval_retrieval")

    lines += ["## Gates", "", "| Gate | Status | Evidence |", "|---|---|---|"]
    if gold:
        g = gold[-1]
        sha = next(iter(g["outputs"].values()))
        lines.append(f"| Gold set frozen before indexing | done | {g['metrics']['rows']} questions "
                     f"({g['metrics']['must_refuse']} must-refuse), sha256 `{sha[:12]}…`, run `{g['run_id']}` |")
    if chunk:
        c = chunk[-1]
        lines.append(f"| Corpus split by page URL | done | {c['metrics']['pages_per_split']}, run `{c['run_id']}` |")
    if retr:
        r = retr[-1]
        ok = r["metrics"]["passes_target"]
        lines.append(f"| recall@5 ≥ 0.7 before training | {'pass' if ok else 'FAIL'} | "
                     f"{r['metrics']['answerable']['recall@5']:.3f}, run `{r['run_id']}` |")
    lines += ["| Baseline (no adapter) scored | not started | |",
              "| SFT data card checks | not started | |",
              "| QLoRA training | not started | |", ""]

    figs = []
    audit = tracking.load("legacy_audit")
    if audit and crawl:
        figs.append(("v1 → v2: coverage", fig_coverage(audit[-1], crawl[-1])))
        figs.append(("v1 → v2: v1's own test queries", fig_old_queries(audit[-1])))
    if chunk:
        figs.append(("Corpus", fig_corpus(chunk[-1])))
    if retr:
        rep = json.loads((EVAL_DIR / "retrieval_report.json").read_text())
        preds = [json.loads(l) for l in (EVAL_DIR / "retrieval_predictions.jsonl").read_text().splitlines()]
        rid = retr[-1]["run_id"]
        figs += [("Retrieval recall@5 by program", fig_recall_by_program(rep, rid)),
                 ("Recall@k", fig_recall_at_k(rep, rid)),
                 ("Top-1 similarity", fig_top1_scores(preds, rid))]
        lines += ["## Retrieval (gold_v1)", "", "| Program | n | R@1 | R@3 | R@5 | R@10 | MRR@10 | fact coverage@5 |", "|---|---|---|---|---|---|---|---|"]
        for p, v in list(rep["per_program"].items()) + [("**all answerable**", rep["answerable"])]:
            lines.append(f"| {LABELS.get(p, p)} | {v['n']} | {v['recall@1']:.2f} | {v['recall@3']:.2f} | "
                         f"{v['recall@5']:.2f} | {v['recall@10']:.2f} | {v['mrr@10']:.2f} | {v['fact_coverage@5']:.2f} |")
        m = rep["must_refuse"]
        lines += ["", f"Must-refuse: {m['n']} questions. {m['n_with_answer_markers']} have answer markers; "
                  f"retrieval_ok on those = {m['retrieval_ok_with_markers']}. The other "
                  f"{m['no_marker_rows_pass_trivially']} (personal assessment, legal strategy) pass retrieval trivially "
                  "and are only meaningful in the generation eval.", ""]
        if rep["misses_at_5"]:
            lines += ["Misses at k=5:", ""]
            lines += [f"- `{x['id']}` {x['question']}" for x in rep["misses_at_5"]]
            lines.append("")
    for title, path in figs:
        lines += [f"### {title}", "", f"![{title}]({path})", ""]

    a = legacy_audit()
    lines += ["## What the v1 artifacts claimed (kept for comparison, not reused)", "",
              f"- `evaluation_report_20251125_150653.json` scores the system **{a['old_overall_score']}** while recording "
              f"**{a['old_test_queries']}** test queries and training progress \"{a['old_training_progress']}\".",
              f"- `rag_evaluation_results_direct.csv` marks all {a['old_csv_queries']} queries successful because "
              "success meant \"a chunk came back\".",
              "- `training_metrics_*.png` plot `2.5·exp(-0.01·step)` + noise from `simulate_training_metrics()`, not a training log.",
              ""]
    if audit:
        f = audit[-1]["metrics"]
        lines += [f"- The v1 crawler saved **{f['v1_pages_crawled']} pages**: {f['v1_pages_by_program']}. "
                  "No study permit, PGWP, work permit, family or citizenship page was in the index.",
                  f"- v1 fine-tuned on **{f['v1_sft_rows']} template rows**; {f['v1_sft_rows_similarity_ge_0_5']} had "
                  f"retrieval similarity ≥ 0.5 (lowest {f['v1_sft_min_similarity']}). No held-out set.",
                  f"- On v1's own test queries, the top chunk was from the right program "
                  f"**{f['v1_top1_right_program']}/{f['compared']}** times (v2: **{f['v2_top1_right_program']}/{f['compared']}**). "
                  f"Source: ledger run `{audit[-1]['run_id']}`.",
                  "", "![v1 loss curve, simulated](../visa_buddy_data/metrics/training_metrics_20251125_145813.png)",
                  "", "*The v1 loss curve above is the original file, kept unchanged as evidence. It was drawn from a formula.*", ""]
    lines += ["## Run ledger", "", "| run | stage | time (UTC) | commit |", "|---|---|---|---|"]
    for r in tracking.load():
        lines.append(f"| `{r['run_id']}` | {r['stage']} | {r['timestamp']} | `{r['git_commit']}`{' (dirty)' if r['git_dirty'] else ''} |")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "PROGRESS.md").write_text("\n".join(lines) + "\n")
    print("wrote reports/PROGRESS.md and", len(figs), "figures")


if __name__ == "__main__":
    main()
