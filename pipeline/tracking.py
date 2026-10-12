"""Append-only run ledger.

Each pipeline stage calls `record(...)` once at the end. A ledger row says which
stage ran, on which commit, which files it read and wrote (with SHA-256), the
config it used, and the metrics it computed. Charts and the progress report are
generated from this file only, so every number on them traces to a run.
"""
import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, RUNS_DIR

LEDGER = RUNS_DIR / "ledger.jsonl"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _files(paths) -> dict:
    out = {}
    for p in paths or []:
        p = Path(p)
        if p.is_file():
            out[str(p.relative_to(ROOT))] = sha256(p)
    return out


def record(stage: str, *, inputs=(), outputs=(), config=None, metrics=None, notes="") -> dict:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": uuid.uuid4().hex[:12],
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "inputs": _files(inputs),
        "outputs": _files(outputs),
        "config": config or {},
        "metrics": metrics or {},
        "notes": notes,
    }
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[ledger] {stage} run {row['run_id']} recorded")
    return row


def load(stage: str | None = None) -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    return [r for r in rows if stage is None or r["stage"] == stage]
