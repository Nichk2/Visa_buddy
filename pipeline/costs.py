"""What the project cost in GPU time, from Modal's own billing data (not estimates).

    modal billing report --start 2026-10-01 --resolution d --show-resources --json > billing.json
    python -m pipeline.costs billing.json          # -> reports/costs.json + a ledger row

The workspace is shared with other projects, so only apps named visabuddy-* are counted.
Colab runs (evals, the final exam, the demo) were on the free tier: $0, listed for completeness.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from . import tracking
from .config import REPORTS_DIR

# What each Modal app did, for the page; the app names are the ones in pipeline/*_modal.py
ROLE = {"visabuddy-teacher": "teacher", "visabuddy-teacher-v4": "teacher",
        "visabuddy-train": "training", "visabuddy-bakeoff": "comparison"}
GPU_RESOURCES = {"H100", "A100-40GB", "A100-80GB", "L4", "T4", "L40S"}


def main(billing: Path):
    rows = [r for r in json.loads(billing.read_text()) if r["description"].startswith("visabuddy")]
    by_role, by_gpu, by_app = defaultdict(float), defaultdict(float), defaultdict(float)
    apps = defaultdict(set)
    for r in rows:
        cost = float(r["cost"])
        by_role[ROLE[r["description"]]] += cost  # KeyError on purpose: a new app must be given a role
        by_app[r["description"]] += cost
        apps[r["description"]].add(r["object_id"])
        if r["resource"] in GPU_RESOURCES:
            by_gpu[r["resource"]] += cost
    total = sum(by_app.values())
    out = {"source": "modal billing report (Modal's billed usage, per app and resource)",
           "currency": "USD", "period": [min(r["interval_start"] for r in rows)[:10], max(r["interval_start"] for r in rows)[:10]],
           "modalTotal": round(total, 2), "gpuOnly": round(sum(by_gpu.values()), 2),
           "byRole": {k: round(v, 2) for k, v in sorted(by_role.items(), key=lambda x: -x[1])},
           "byGpu": {k: round(v, 2) for k, v in sorted(by_gpu.items(), key=lambda x: -x[1])},
           "byApp": {k: {"usd": round(v, 2), "runs": len(apps[k])} for k, v in sorted(by_app.items(), key=lambda x: -x[1])},
           "paidWithCredits": True,  # Modal's monthly free credits covered all of it
           "colab": {"usd": 0, "used_for": "evaluation, the final exam and the demo (free T4)"}}
    path = REPORTS_DIR / "costs.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    tracking.record("costs", inputs=[billing], outputs=[path], metrics=out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
