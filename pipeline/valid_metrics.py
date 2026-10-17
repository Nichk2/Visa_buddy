"""Score generated answers on valid.jsonl, used to pick the checkpoint (not train loss).

valid rows have a reference answer, not gold expected_facts, so "fact coverage" here is
the share of the reference answer's grounded content (content tokens that also appear in
the row's context) that the generated answer reproduces. Refusal rows score on whether
the model refused. Kept free of heavy imports so the Modal training job can use it.
"""
import re

from .grade import grade
from .prompt import REFUSAL
from .text import norm, tokens

_SRC_RE = re.compile(r"\(Source:[^)]*\)|https?://\S+")


def _ctx_text(row):
    return row["input"].split("QUESTION:")[0]


def grounded_tokens(text, ctx):
    ctx_t = set(tokens(ctx))
    return {t for t in tokens(_SRC_RE.sub(" ", text)) if t in ctx_t}


def score_valid(rows, outputs):
    ans_cov, cites, over_refusals, ref_ok, unsupported = [], [], 0, [], 0
    n_ans = 0
    for row, out in zip(rows, outputs):
        refused = norm(REFUSAL) in norm(out)
        if row["kind"] == "answer":
            n_ans += 1
            ctx = _ctx_text(row)
            ref = grounded_tokens(row["output"], ctx)
            got = grounded_tokens(out, ctx)
            ans_cov.append(len(ref & got) / len(ref) if ref else 0.0)
            if refused:
                over_refusals += 1
            else:
                hits = [{"url": u, "text": ctx} for u in row["context_urls"]]
                g = grade(out, hits, {"expected_facts": [], "must_refuse": False})
                cites.append(g["citation_hit"])
                unsupported += g["unsupported"]
        else:
            ref_ok.append(refused)
    return {"fact_coverage": round(sum(ans_cov) / len(ans_cov), 4) if ans_cov else None,
            "cite_when_answering": round(sum(cites) / len(cites), 4) if cites else None,
            "over_refusal_rate": round(over_refusals / n_ans, 4) if n_ans else None,
            "refusal_accuracy": round(sum(ref_ok) / len(ref_ok), 4) if ref_ok else None,
            "unsupported_rate": round(unsupported / max(len(cites), 1), 4),
            "n_answer": n_ans, "n_refusal": len(ref_ok)}


# --- v4: two tasks (extract quotes, answer from quotes) ---------------------------------
def _f1(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return 0.0 if not inter else 2 * inter / (len(ta) + len(tb))


def score_valid_v4(rows, outputs):
    """Checkpoint selection for v4 (fixed before training). Three parts, averaged:
    quote_match: extract rows with target quotes, best token-F1 of each target quote against
      the predicted quotes (0 if the output isn't JSON or refuses)
    refusal_accuracy: extract rows whose target is {"quotes": []} or {"refuse": true}
    answer_coverage: answer rows, share of the reference answer's grounded words reproduced"""
    import json as _json
    import re as _re
    qm, ra, ac, bad_json, false_refusals = [], [], [], 0, 0
    for row, out in zip(rows, outputs):
        if row["task"] == "extract":
            target = _json.loads(row["output"])
            m = _re.search(r"\{.*\}", out, flags=_re.S)
            try:
                pred = _json.loads(m.group(0)) if m else None
            except _json.JSONDecodeError:
                pred = None
            if pred is None:
                bad_json += 1
                pred = {}
            pq = [q["text"] if isinstance(q, dict) else str(q) for q in pred.get("quotes") or []]
            if target.get("quotes"):
                if not pq:
                    false_refusals += 1
                qm.append(sum(max((_f1(t["text"], p) for p in pq), default=0.0) for t in target["quotes"]) / len(target["quotes"]))
            elif target.get("refuse"):
                ra.append(pred.get("refuse") is True)
            else:
                ra.append(pred.get("refuse") is not True and not pq)
        else:
            quotes = row["input"].split("QUESTION:")[0]
            ref = grounded_tokens(row["output"], quotes)
            got = grounded_tokens(out, quotes)
            ac.append(len(ref & got) / len(ref) if ref else 0.0)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    parts = [mean(qm), mean(ra), mean(ac)]
    return {"v4_valid_score": round(sum(p for p in parts if p is not None) / max(1, sum(p is not None for p in parts)), 4),
            "quote_match": parts[0], "refusal_accuracy": parts[1], "answer_coverage": parts[2],
            "false_refusals": false_refusals, "bad_json": bad_json,
            "fact_coverage": parts[0]}  # alias: the training job selects on "fact_coverage"
