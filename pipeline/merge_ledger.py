"""Merge ledger rows produced on Colab or Modal into the local ledger.

    python -m pipeline.merge_ledger [runs/ledger.colab.jsonl]

The Colab notebook exports its ledger as runs/ledger.colab.jsonl so unzipping never
overwrites the local one. Rows are unioned by run_id and kept in timestamp order.
"""
import json
import sys
from pathlib import Path

from .tracking import LEDGER


def rows(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main(other: Path):
    local = rows(LEDGER)
    merged = {r["run_id"]: r for r in local + rows(other)}
    out = sorted(merged.values(), key=lambda r: r["timestamp"])
    LEDGER.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"added {len(out) - len(local)} runs from {other}; ledger now has {len(out)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else LEDGER.with_name("ledger.colab.jsonl"))
