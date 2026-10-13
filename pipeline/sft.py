"""Build supervised fine-tuning data (step 5).

    python -m pipeline.sft select      # pick source chunks, split pages train/valid -> work/questions_requests.jsonl
    modal run pipeline/teacher_modal.py --phase questions
    python -m pipeline.sft requests    # parse questions, leakage filter, build contexts -> work/answers_requests.jsonl
    modal run pipeline/teacher_modal.py --phase answers
    python -m pipeline.sft gate        # teacher on gold_v2 vs thresholds set before the run
    python -m pipeline.sft build       # filters, refusal rows, train/valid, DATA_CARD.md

Every row uses the serving prompt (pipeline.prompt) over contexts from the serving
retriever, restricted to training pages: gold source pages (any gold version) never
appear in a training input or as the source of a target. valid.jsonl is split by page
URL and shares no source page with train.
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from . import tracking
from .config import DATA, EVAL_DIR, GOLD_V2, PROGRAMS, TOP_K
from .grade import grade
from .prompt import REFUSAL, REFUSAL_OUTPUT, SYSTEM, build_messages, format_context
from .retriever import Retriever
from .text import norm, tokens

SFT = DATA / "sft"
WORK = SFT / "work"
SEED = 13
PER_PROGRAM = 80          # source chunks asked about per program
VALID_PAGE_FRAC = 0.12    # share of source pages held out for validation
MIN_CHUNK_CHARS = 400
GOLD_SIM_MAX = 0.85       # a training question this close to any gold question is dropped
DUP_SIM_MAX = 0.95        # near-duplicate training questions are dropped
GROUNDED_MIN = 0.85       # share of answer content tokens that must appear in the context
REFUSAL_SHARE = 0.27      # target share of refusal rows (data card requires >= 0.20)

# Thresholds for the teacher on gold_v2 (grader v2), fixed before the run. The teacher must
# clearly beat the untuned student on what we're teaching, or its rows are not used.
GATE = {"citation_hit_min": 0.90, "false_refusal_rate_max": 0.10, "refusal_correct_on_must_refuse_min": 0.90,
        "unsupported_max": 0.05, "fact_coverage_min": 0.585}  # 0.585 = baseline on gold_v2
# The first gate run (32dcaaaba95d) applied the citation and refusal thresholds to the v1 metric
# definitions, which count a refusal as a citation miss and count refusing when the sources lack
# the answer as an error. It failed on that. With the user's approval the same thresholds are
# applied to the corrected (metrics v2) definitions; the failed run stays in the ledger.
GATE_METRIC = {"citation_hit_min": "cite_when_answering", "false_refusal_rate_max": "over_refusal_rate"}

# Out-of-scope topics for refusal questions. Sub-topics used by gold must_refuse questions
# (OINP, AAIP, Arrima, refugee claims at an airport, CRS cut-off, citizenship/PGP fees,
# India processing times, judicial review, NOC misrepresentation) are left out on purpose.
OOS_TOPICS = [
    "the British Columbia Provincial Nominee Program", "the Saskatchewan Immigrant Nominee Program",
    "the Manitoba Provincial Nominee Program", "the Nova Scotia Nominee Program",
    "getting a US visa or green card", "UK or Australian visas",
    "whether their own work permit or visitor visa application will be approved (asking for a personal prediction)",
    "how to hide a previous visa refusal or criminal charge on an application",
    "which legal arguments to use to overturn a refused spousal sponsorship",
    "provincial health insurance coverage after arriving in Canada", "finding a job or writing a resume in Canada",
    "Canadian income tax filing details", "the Atlantic Immigration Program", "the Start-up Visa program",
    "caregiver immigration programs", "privately sponsoring a refugee family",
]

Q_SYSTEM = """You write realistic questions for a Canadian immigration help desk.
You receive one passage from an official IRCC web page. Write ONE question that a real applicant could ask and that this passage answers directly and completely.
Rules:
- Write as the applicant ("Can I...", "How much...", "Do I need...").
- Do not mention the passage, the page or a source.
- Name the program if the passage is about one.
- If the passage has no factual content an applicant would ask about (navigation, link lists, headers), use null.
Reply with ONLY a JSON object: {"question": "..."} or {"question": null}"""

OOS_SYSTEM = """You write realistic questions people send to a Canadian immigration help desk.
Reply with ONLY a JSON object: {"questions": ["...", "..."]}"""


def _jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _parse_json(text):
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def _gold():
    return _jsonl(GOLD_V2)


# ---------------------------------------------------------------- select
def select():
    ret = Retriever()
    rng = random.Random(SEED)
    pool = defaultdict(list)
    for c in ret.chunks:
        if c["split"] == "train_eligible" and len(c["text"]) >= MIN_CHUNK_CHARS and c["program"] in PROGRAMS:
            pool[c["program"]].append(c)
    chosen = []
    for p in PROGRAMS:
        if p in pool:
            chosen += rng.sample(pool[p], min(PER_PROGRAM, len(pool[p])))
    pages = sorted({c["url"] for c in chosen})
    rng.shuffle(pages)
    valid_pages = set(pages[: round(len(pages) * VALID_PAGE_FRAC)])
    reqs = []
    for c in chosen:
        user = f"Page title: {c['title']}\nSection: {c['section']}\nPassage:\n<<<\n{c['text']}\n>>>"
        reqs.append({"req_id": f"q:{c['chunk_id']}", "kind": "chunk_question", "chunk_id": c["chunk_id"],
                     "url": c["url"], "program": c["program"], "split": "valid" if c["url"] in valid_pages else "train",
                     "messages": [{"role": "system", "content": Q_SYSTEM}, {"role": "user", "content": user}]})
    for i, t in enumerate(OOS_TOPICS):
        user = (f"Write 6 different questions someone might ask a Canadian immigration assistant about {t}. "
                "Vary the wording and detail. Questions only, no answers.")
        reqs.append({"req_id": f"oos:{i}", "kind": "oos_question", "topic": t,
                     "messages": [{"role": "system", "content": OOS_SYSTEM}, {"role": "user", "content": user}]})
    _write(WORK / "questions_requests.jsonl", reqs)
    (WORK / "valid_pages.json").write_text(json.dumps(sorted(valid_pages), indent=1))
    by_prog = Counter(c["program"] for c in chosen)
    print(f"{len(chosen)} chunks from {len(pages)} pages ({len(valid_pages)} valid pages), {len(OOS_TOPICS)} OOS topics")
    tracking.record("sft_select", outputs=[WORK / "questions_requests.jsonl", WORK / "valid_pages.json"],
                    config={"seed": SEED, "per_program": PER_PROGRAM, "valid_page_frac": VALID_PAGE_FRAC,
                            "min_chunk_chars": MIN_CHUNK_CHARS, "oos_topics": OOS_TOPICS},
                    metrics={"chunks": len(chosen), "chunks_per_program": dict(by_prog), "pages": len(pages),
                             "valid_pages": len(valid_pages)})


# ---------------------------------------------------------------- requests
class _Embedder:
    def __init__(self, ret: Retriever):
        self.m = ret.model

    def __call__(self, texts):
        return self.m.encode(texts, normalize_embeddings=True, batch_size=64)


def requests_():
    ret = Retriever()
    emb = _Embedder(ret)
    qreqs = {r["req_id"]: r for r in _jsonl(WORK / "questions_requests.jsonl")}
    outs = _jsonl(WORK / "questions_teacher.jsonl")
    valid_pages = set(json.loads((WORK / "valid_pages.json").read_text()))
    chunk_by_id = {c["chunk_id"]: c for c in ret.chunks}
    gold = _gold()
    gold_q = [g["question"] for g in gold]
    gold_norm = {norm(q) for q in gold_q}
    gold_emb = emb(gold_q)

    drops = Counter()
    cands = []  # (question, meta)
    for o in outs:
        r = qreqs[o["req_id"]]
        obj = _parse_json(o["raw"])
        if r["kind"] == "chunk_question":
            q = (obj or {}).get("question")
            if not q or not isinstance(q, str):
                drops["teacher_null_or_unparsed"] += 1
                continue
            cands.append((q.strip(), {"kind": "answer", "chunk_id": r["chunk_id"], "url": r["url"],
                                      "program": r["program"], "split": r["split"]}))
        else:
            qs = (obj or {}).get("questions") or []
            if not qs:
                drops["oos_unparsed"] += 1
            for j, q in enumerate(qs[:6]):
                if isinstance(q, str) and q.strip():
                    h = int(re.sub(r"\D", "", o["req_id"]) or 0) * 10 + j
                    cands.append((q.strip(), {"kind": "oos", "topic": r["topic"], "program": "out_of_scope",
                                              "split": "valid" if h % 8 == 0 else "train"}))

    # leakage + near-duplicate filters
    E = emb([q for q, _ in cands])
    keep, kept_emb = [], []
    for (q, meta), e in zip(cands, E):
        if norm(q) in gold_norm:
            drops["exact_gold_question"] += 1
            continue
        if float(np.max(gold_emb @ e)) >= GOLD_SIM_MAX:
            drops["near_gold_question"] += 1
            continue
        if kept_emb and float(np.max(np.array(kept_emb) @ e)) >= DUP_SIM_MAX:
            drops["near_duplicate"] += 1
            continue
        keep.append((q, meta))
        kept_emb.append(e)

    def allow_for(split, *, program=None, not_program=None, not_url=None):
        def f(c):
            if c["split"] != "train_eligible":
                return False
            if split == "train" and c["url"] in valid_pages:
                return False
            if program and c["program"] != program:
                return False
            if not_program and c["program"] == not_program:
                return False
            return not (not_url and c["url"] == not_url)
        return f

    reqs = []
    n = 0
    for q, meta in keep:
        n += 1
        base = {"question": q, **meta}
        if meta["kind"] == "oos":
            hits = ret.search(q, TOP_K, allow=allow_for(meta["split"]))
            reqs.append(_answer_req(f"oos:{n}", "refusal_out_of_scope", base, hits))
            continue
        src = chunk_by_id[meta["chunk_id"]]
        hits = ret.search(q, TOP_K, allow=allow_for(meta["split"]))
        if src["chunk_id"] not in {h["chunk_id"] for h in hits}:
            drops["source_chunk_not_retrieved"] += 1
        else:
            reqs.append(_answer_req(f"ans:{n}", "answer", base, hits))
        # refusal candidates built from the same question with the answer's page removed
        reqs.append(_answer_req(f"wrongprog:{n}", "refusal_wrong_program", base,
                                ret.search(q, TOP_K, allow=allow_for(meta["split"], not_program=src["program"]))))
        reqs.append(_answer_req(f"removed:{n}", "refusal_removed", base,
                                ret.search(q, TOP_K, allow=allow_for(meta["split"], program=src["program"], not_url=src["url"]))))
    # gate: teacher answers gold_v2 with the exact serving retrieval (never used for training)
    for g in gold:
        reqs.append(_answer_req(f"gate:{g['id']}", "gate", {"question": g["question"], "gold_id": g["id"]},
                                ret.search(g["question"], TOP_K)))
    _write(WORK / "answers_requests.jsonl", reqs)
    kinds = Counter(r["kind"] for r in reqs)
    print(f"{len(keep)} questions kept; drops {dict(drops)}; answer requests {dict(kinds)}")
    tracking.record("sft_requests", inputs=[WORK / "questions_teacher.jsonl", GOLD_V2], outputs=[WORK / "answers_requests.jsonl"],
                    config={"gold_sim_max": GOLD_SIM_MAX, "dup_sim_max": DUP_SIM_MAX, "top_k": TOP_K},
                    metrics={"questions_kept": len(keep), "drops": dict(drops), "requests": dict(kinds)})


def _answer_req(req_id, kind, base, hits):
    return {"req_id": req_id, **{k: v for k, v in base.items() if k != "kind"}, "kind": kind,
            "context": [{k: h[k] for k in ("chunk_id", "url", "title", "program", "text")} for h in hits],
            "messages": build_messages(base["question"], hits)}


# ---------------------------------------------------------------- gate
def _gate_rows():
    reqs = {r["req_id"]: r for r in _jsonl(WORK / "answers_requests.jsonl")}
    outs = {o["req_id"]: o["raw"] for o in _jsonl(WORK / "answers_teacher.jsonl")}
    gold = {g["id"]: g for g in _gold()}
    rows = []
    for rid, r in reqs.items():
        if r["kind"] == "gate":
            g = gold[r["gold_id"]]
            rows.append({"id": g["id"], "program": g["program"], "must_refuse": g["must_refuse"],
                         "question": g["question"], "answer": outs[rid], "grade": grade(outs[rid], r["context"], g)})
    return rows


def gate():
    from .eval_answers import summarize
    rows = _gate_rows()
    s = summarize(rows)["overall"]
    checks = {GATE_METRIC["citation_hit_min"]: s[GATE_METRIC["citation_hit_min"]] >= GATE["citation_hit_min"],
              GATE_METRIC["false_refusal_rate_max"]: s[GATE_METRIC["false_refusal_rate_max"]] <= GATE["false_refusal_rate_max"],
              "refusal_correct_on_must_refuse": s["refusal_correct_on_must_refuse"] >= GATE["refusal_correct_on_must_refuse_min"],
              "unsupported": s["unsupported"] <= GATE["unsupported_max"],
              "fact_coverage": s["fact_coverage"] >= GATE["fact_coverage_min"]}
    passed = all(checks.values())
    meta = json.loads((WORK / "answers_teacher_meta.json").read_text())
    rep = {"teacher": meta["teacher"], "gold": GOLD_V2.name, "grader_version": 2, "metrics_version": 2,
           "thresholds": GATE, "threshold_applies_to": GATE_METRIC, "supersedes_failed_gate_run": "32dcaaaba95d",
           "overall": s, "checks": checks, "passed": passed}
    _write(EVAL_DIR / "teacher_gate_predictions.jsonl", rows)
    (EVAL_DIR / "teacher_gate_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    tracking.record("teacher_gate", inputs=[WORK / "answers_teacher.jsonl", GOLD_V2],
                    outputs=[EVAL_DIR / "teacher_gate_report.json", EVAL_DIR / "teacher_gate_predictions.jsonl"],
                    config={"thresholds": GATE, "threshold_applies_to": GATE_METRIC, "metrics_version": 2, **meta},
                    metrics={"overall": s, "checks": checks, "passed": passed},
                    notes="Re-check of failed gate 32dcaaaba95d under corrected metric definitions; thresholds unchanged.")
    return passed


# ---------------------------------------------------------------- build
_SRC_RE = re.compile(r"\(Source:[^)]*\)|https?://\S+")


def grounded_share(answer, ctx_text):
    body = _SRC_RE.sub(" ", answer)
    toks = tokens(body)
    ctx = set(tokens(ctx_text))
    return sum(t in ctx for t in toks) / len(toks) if toks else 0.0


def build():
    rep = json.loads((EVAL_DIR / "teacher_gate_report.json").read_text())
    if not rep["passed"]:
        sys.exit("teacher gate failed: its rows are not used (see eval/teacher_gate_report.json)")
    reqs = _jsonl(WORK / "answers_requests.jsonl")
    outs = {o["req_id"]: o["raw"].strip() for o in _jsonl(WORK / "answers_teacher.jsonl")}
    gold_urls = {u for g in _gold() for u in g["source_urls"]}
    drops = Counter()
    answers, refusals = [], []
    for r in reqs:
        if r["kind"] == "gate":
            continue
        out = outs[r["req_id"]]
        ctx_text = "\n".join(c["text"] for c in r["context"])
        ctx_urls = {c["url"] for c in r["context"]}
        assert not ctx_urls & gold_urls, f"{r['req_id']}: gold page in a training context"
        teacher_refused = norm(REFUSAL) in norm(out)
        row = {"id": r["req_id"], "kind": r["kind"], "split": r["split"], "program": r["program"],
               "source_url": r.get("url"), "question": r["question"], "context_urls": sorted(ctx_urls),
               "instruction": SYSTEM, "input": r["messages"][1]["content"]}
        if r["kind"] == "answer":
            g = grade(out, r["context"], {"expected_facts": [], "must_refuse": False})
            if teacher_refused:
                drops["answer:teacher_refused"] += 1
            elif not g["citation_hit"]:
                drops["answer:no_valid_citation"] += 1
            elif g["unsupported"]:
                drops["answer:number_not_in_context"] += 1
            elif grounded_share(out, ctx_text) < GROUNDED_MIN:
                drops["answer:not_grounded"] += 1
            elif len(out) > 1500:
                drops["answer:too_long"] += 1
            else:
                answers.append({**row, "output": out})
        else:
            # refusal targets: the teacher must also have refused this exact context
            if not teacher_refused:
                drops[f"{r['kind']}:teacher_answered"] += 1
            else:
                refusals.append({**row, "output": REFUSAL_OUTPUT})

    # balance: refusals ~REFUSAL_SHARE of the final set, mixing the three kinds
    rng = random.Random(SEED)
    want = round(len(answers) * REFUSAL_SHARE / (1 - REFUSAL_SHARE))
    by_kind = defaultdict(list)
    for r in refusals:
        by_kind[r["kind"]].append(r)
    for v in by_kind.values():
        rng.shuffle(v)
    picked = []
    while len(picked) < want and any(by_kind.values()):
        for k in sorted(by_kind):
            if by_kind[k] and len(picked) < want:
                picked.append(by_kind[k].pop())
    rows = answers + picked
    rng.shuffle(rows)
    train = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "valid"]
    _write(SFT / "train.jsonl", train)
    _write(SFT / "valid.jsonl", valid)
    card = data_card(train, valid, drops, len(refusals))
    (SFT / "DATA_CARD.md").write_text(card["markdown"])
    print(card["markdown"][:3000])
    tracking.record("sft_build", inputs=[WORK / "answers_requests.jsonl", WORK / "answers_teacher.jsonl", GOLD_V2],
                    outputs=[SFT / "train.jsonl", SFT / "valid.jsonl", SFT / "DATA_CARD.md"],
                    config={"grounded_min": GROUNDED_MIN, "refusal_share_target": REFUSAL_SHARE, "seed": SEED},
                    metrics=card["metrics"])


def data_card(train, valid, drops, refusal_pool):
    gold = _gold()
    gold_urls = {u for g in gold for u in g["source_urls"]}
    gold_qs = {norm(g["question"]) for g in gold}
    allrows = train + valid
    checks = {
        "no_gold_question_in_any_row": not any(norm(r["question"]) in gold_qs or any(q in norm(r["output"]) for q in gold_qs) for r in allrows),
        "no_gold_source_url_as_source": not any(r["source_url"] in gold_urls for r in allrows if r["source_url"]),
        "no_gold_url_in_any_context": not any(set(r["context_urls"]) & gold_urls for r in allrows),
        "refusal_share_train_ge_0_20": sum(r["kind"] != "answer" for r in train) / max(len(train), 1) >= 0.20,
        "valid_shares_no_source_url_with_train": not ({r["source_url"] for r in train if r["source_url"]}
                                                      & {r["source_url"] for r in valid if r["source_url"]}),
    }
    kinds = lambda rs: dict(Counter(r["kind"] for r in rs))
    m = {"train": len(train), "valid": len(valid), "train_kinds": kinds(train), "valid_kinds": kinds(valid),
         "train_programs": dict(Counter(r["program"] for r in train)),
         "refusal_share_train": round(sum(r["kind"] != "answer" for r in train) / max(len(train), 1), 3),
         "drops": dict(drops), "refusal_candidates": refusal_pool, "checks": checks, "all_checks_pass": all(checks.values())}
    rng = random.Random(SEED + 1)
    sample = rng.sample(train, min(20, len(train)))
    md = ["# SFT data card (Visa Buddy v2)", "",
          "Generated by `python -m pipeline.sft build`. Teacher: Mistral Small 3.2 24B (see eval/teacher_gate_report.json).", "",
          f"- train rows: **{len(train)}** {kinds(train)}", f"- valid rows: **{len(valid)}** {kinds(valid)}",
          f"- refusal share (train): **{m['refusal_share_train']}** (required >= 0.20)",
          f"- train rows per program: {m['train_programs']}", f"- dropped by filters: {dict(drops)}", "",
          "## Checks", ""] + [f"- [{'x' if v else ' '}] {k}" for k, v in checks.items()] + [
          "", "## Manual review (before training)", "",
          "The 20 rows below were read against their inputs: each output sentence was compared with the source text "
          "in the row's context, and claims a sentence-matcher couldn't place (\"15 of 20 questions\", bankruptcy rule, "
          "NYIT Master of Data Science, online visitor record, stateless people born to a Canadian parent) were found "
          "in the context. No output adds a fee, timeline or eligibility rule that isn't in its input. Refusal rows use "
          "one fixed output and are kept only when the teacher also refused that exact context.", "",
          "## Known weaknesses", "",
          "- PGWP and Express Entry have few training rows: most of their pages are gold source pages and are held out.",
          "- Questions were written by the teacher from single chunks, so they are more specific than typical user questions.",
          "- Answer rows require the source chunk in the top 5; questions the retriever can't serve are not trained on.",
          "", "## 20 random train rows", ""] + [
          ]
    for r in sample:
        md += [f"### {r['id']} ({r['kind']}, {r['program']})", f"**Q:** {r['question']}", "",
               f"**Context pages:** {', '.join(u.split('/services/')[-1] for u in r['context_urls'])}", "",
               f"**Output:** {r['output']}", ""]
    return {"metrics": m, "markdown": "\n".join(md) + "\n"}


if __name__ == "__main__":
    {"select": select, "requests": requests_, "gate": gate, "build": build}[sys.argv[1]]()
