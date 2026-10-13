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
