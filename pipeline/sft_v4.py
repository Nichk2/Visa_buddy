"""Phase 5: fine-tuning data for v4 (Qwen3-4B learns the v4 pipeline's two steps).

    python -m pipeline.sft_v4 select     # question requests for the teacher -> sft_v4/work/questions_todo.jsonl
    caffeinate -i modal run --detach pipeline/teacher_modal.py --phase questions --data v4
    python -m pipeline.sft_v4 items      # leakage filters + hybrid retrieval (training pages only)
    caffeinate -i modal run --detach pipeline/teacher_v4_modal.py   # teacher runs the v4 pipeline (+ gate)
    python -m pipeline.sft_v4 gate       # teacher vs thresholds fixed before the run
    python -m pipeline.sft_v4 build      # train/valid rows for both steps, DATA_CARD.md

Targets come from the teacher running the real v4 pipeline (extract -> verify -> answer -> verify),
so every extract target is verbatim canada.ca text and every answer passed the app's own checks.
Aimed at what the untuned Qwen3-4B still got wrong (reports/V4_PLAN.md, phase 3):
  - situation rows: passages with several "If you..." cases; the question fits one case (y02)
  - no-quote rows: the answer's page is out of reach; target {"quotes": []} (rf10, clean refusals)
  - policy rows: misrepresentation / legal strategy / predictions; target {"refuse": true}
  - plenty of answerable rows, to cut over-refusal (0.13)
Every page and question of every eval set (incl. the sealed final exam) is held out.
"""
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

from . import tracking
from .config import DATA, EVAL_DIR, GOLD_FILES, TOP_K
from .sft import OOS_SYSTEM, OOS_TOPICS, MISREP_TOPICS, Q_SYSTEM, _cache, _parse_json
from .text import norm, tokens

SFT = DATA / "sft_v4"
WORK = SFT / "work"
INDEX_V2 = DATA / "index_v2"
SEED = 17
NEW_PAGE_CHUNKS = 160     # questions on corpus_v2 pages that no earlier SFT version used
SITUATION_CHUNKS = 160    # passages with several "If you..." cases
SIM_MAX = 0.80            # same leakage threshold as v3, against every eval question
VALID_FRAC = 0.12         # valid split by page
REFUSAL_SHARE = 0.25      # of extract rows: no-quote + policy refusals
GROUNDED_MIN = 0.85       # answer words that must appear in its quotes

# Teacher gate, fixed before the run: the teacher, running the same v4 pipeline on the 126 eval
# questions, must beat the untuned student (bake-off run 3) or its rows are not used.
GATE = {"wrong_rules_shown_max": 1, "must_refuse_min": 31, "gold_fact_coverage_min": 0.762,
        "gold_over_refusal_max": 0.10}

SITUATION_SYSTEM = """You write realistic questions for a Canadian immigration help desk.
The passage below gives DIFFERENT rules for DIFFERENT situations (for example "If you transferred between schools..." and "If you transferred from a non-DLI school...").
Write ONE question from an applicant in ONE specific situation, whose correct answer depends on reading the rule for that situation and not the others. Describe the situation in the applicant's own words.
Do not mention the passage. Reply with ONLY a JSON object: {"question": "..."} or {"question": null} if the passage has no such cases."""
_COND = re.compile(r"^(?:##\s*)?If you(?:'|’)?(?:re| are| have| were|ve)?\b", re.I | re.M)


def _jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _sha(m):
    return hashlib.sha256(json.dumps(m, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _eval_rows():
    return [r for p in GOLD_FILES if p.exists() for r in _jsonl(p)]


def _chunks():
    return _jsonl(INDEX_V2 / "chunks.jsonl")


def _valid(url):  # deterministic page-level split
    return int(hashlib.sha1(url.encode()).hexdigest(), 16) % 100 < VALID_FRAC * 100


# ---------------------------------------------------------------- select
def select():
    rng = random.Random(SEED)
    chunks = [c for c in _chunks() if c["split"] == "train_eligible" and len(c["text"]) >= 400]
    v1_urls = {d["url"] for d in _jsonl(DATA / "corpus" / "docs.jsonl")}
    new = [c for c in chunks if c["url"] not in v1_urls]
    situ = [c for c in chunks if len(_COND.findall(c["text"])) >= 2]
    pick_new = rng.sample(new, min(NEW_PAGE_CHUNKS, len(new)))
    taken = {c["chunk_id"] for c in pick_new}
    pick_situ = rng.sample([c for c in situ if c["chunk_id"] not in taken], min(SITUATION_CHUNKS, len(situ)))
    reqs = []
    for kind, sysmsg, picked in (("chunk_question", Q_SYSTEM, pick_new), ("situation_question", SITUATION_SYSTEM, pick_situ)):
        for c in picked:
            user = f"Page title: {c['title']}\nSection: {c['section']}\nPassage:\n<<<\n{c['text']}\n>>>"
            reqs.append({"req_id": f"{kind}:{c['chunk_id']}", "kind": kind, "url": c["url"], "program": c["program"],
                         "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]})
    _write(WORK / "questions_todo.jsonl", reqs)
    print(f"{len(reqs)} question requests ({len(pick_new)} new-page, {len(pick_situ)} situation)")
    tracking.record("sft_v4_select", outputs=[WORK / "questions_todo.jsonl"],
                    config={"seed": SEED, "new_page_chunks": NEW_PAGE_CHUNKS, "situation_chunks": SITUATION_CHUNKS},
                    metrics={"requests": len(reqs), "new_page": len(pick_new), "situation": len(pick_situ)})


# ---------------------------------------------------------------- items
def items():
    """Collect questions (new + reused from v3), filter leakage, retrieve contexts."""
    from .retriever import Retriever
    ret = Retriever(INDEX_V2, "hybrid")
    split = {c["url"]: c["split"] for c in ret.chunks}
    cands = []
    # new questions from this version's teacher run
    reqs = {r["req_id"]: r for r in _jsonl(WORK / "questions_todo.jsonl")}
    for o in _jsonl(WORK / "questions_todo_teacher.jsonl"):
        q = (_parse_json(o["raw"]) or {}).get("question")
        r = reqs[o["req_id"]]
        if isinstance(q, str) and q.strip():
            cands.append({"question": q.strip(), "kind": "answer" if r["kind"] == "chunk_question" else "situation",
                          "url": r["url"], "program": r["program"]})
    # reused questions from v3's teacher run (text and source page only; contexts are rebuilt here)
    for r in _jsonl(DATA / "sft_v3" / "work" / "answers_requests.jsonl"):
        k = {"answer": "answer", "refusal_misrepresentation": "policy", "refusal_out_of_scope": "policy"}.get(r["kind"])
        if k:
            cands.append({"question": r["question"], "kind": k, "url": r.get("url"), "program": r["program"]})
    # leakage + duplicates
    ev = sorted({r["question"] for r in _eval_rows()})
    e_ev = ret.model.encode(ev, normalize_embeddings=True)
    e_c = ret.model.encode([c["question"] for c in cands], normalize_embeddings=True, batch_size=64)
    keep, kept_e, drops = [], [], Counter()
    for c, e in zip(cands, e_c):
        if c["url"] and split.get(c["url"]) != "train_eligible":
            drops["source_page_not_trainable"] += 1
            continue
        if float(np.max(e_ev @ e)) >= SIM_MAX:
            drops["near_eval_question"] += 1
            continue
        if kept_e and float(np.max(np.array(kept_e) @ e)) >= 0.95:
            drops["near_duplicate"] += 1
            continue
        keep.append(c)
        kept_e.append(e)
    train_only = lambda c: c["split"] == "train_eligible"
    out = []
    for n, c in enumerate(keep):
        valid = _valid(c["url"]) if c["url"] else (n % 8 == 0)
        allow = (lambda ch: train_only(ch) and (_valid(ch["url"]) == valid))  # contexts never cross the split
        hits = ret.search(c["question"], TOP_K, allow=allow)
        base = {**c, "split": "valid" if valid else "train"}
        slim = lambda hs: [{k: h[k] for k in ("url", "title", "text", "chunk_id")} for h in hs]
        if c["kind"] == "policy":
            out.append({**base, "item_id": f"policy:{n}", "hits": slim(hits)})
            continue
        if c["url"] in {h["url"] for h in hits}:
            out.append({**base, "item_id": f"{c['kind']}:{n}", "hits": slim(hits)})
        else:
            drops["source_page_not_retrieved"] += 1
        # a no-quote twin: same question, the source page out of reach
        far = ret.search(c["question"], TOP_K, allow=lambda ch: allow(ch) and ch["url"] != c["url"])
        out.append({**base, "kind": "no_quote", "item_id": f"noquote:{n}", "hits": slim(far)})
    _write(WORK / "pipeline_items.jsonl", out)
    print(f"{len(keep)} questions kept; drops {dict(drops)}; items {dict(Counter(o['kind'] for o in out))}")
    tracking.record("sft_v4_items", inputs=[WORK / "questions_todo_teacher.jsonl"], outputs=[WORK / "pipeline_items.jsonl"],
                    config={"sim_max": SIM_MAX, "retriever": "hybrid index_v2, training pages only"},
                    metrics={"questions": len(keep), "drops": dict(drops), "items": dict(Counter(o["kind"] for o in out))})


# ---------------------------------------------------------------- gate
def _sfx(teacher):
    return "" if teacher == "mistral-small-3.2" else f"_{teacher}"


def gate(teacher: str = "mistral-small-3.2"):
    from .bakeoff import SETS
    from .eval_answers import summarize
    from .grade import grade
    items_ = {(r["set"], r["id"]): r for r in _jsonl(EVAL_DIR / "bakeoff" / "items.jsonl")}
    gold = {(s, g["id"]): g for s in SETS for g in _jsonl(EVAL_DIR / f"{s}.jsonl")}
    rows = []
    for r in _jsonl(WORK / f"teacher_gate_raw{_sfx(teacher)}.jsonl"):
        key = (r["set"], r["id"])
        g = gold[key]
        rows.append({**r, "program": g["program"], "must_refuse": g["must_refuse"], "grade": grade(r["response"], items_[key]["hits"], g)})
    gold_rows = [x for x in rows if x["set"] == "gold_v2"]
    s = summarize(gold_rows)["overall"]
    wrong = sum(bool(x["grade"].get("wrong_verdict")) or bool(x["grade"].get("contradicts_source")) for x in rows)
    refused = sum(x["grade"]["refused"] for x in rows if x["must_refuse"])
    checks = {"wrong_rules_shown": wrong <= GATE["wrong_rules_shown_max"], "must_refuse": refused >= GATE["must_refuse_min"],
              "gold_fact_coverage": s["fact_coverage"] >= GATE["gold_fact_coverage_min"],
              "gold_over_refusal": s["over_refusal_rate"] <= GATE["gold_over_refusal_max"]}
    rep = {"teacher": teacher, "thresholds": GATE, "wrong_rules_shown": wrong, "must_refuse_refused": refused, "gold": s,
           "checks": checks, "passed": all(checks.values())}
    (WORK / f"teacher_gate_report{_sfx(teacher)}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("wrong_rules_shown", "must_refuse_refused", "checks", "passed")}, indent=1),
          "\ngold fact coverage", s["fact_coverage"], "over-refusal", s["over_refusal_rate"])
    tracking.record("teacher_gate:v4", inputs=[WORK / f"teacher_gate_raw{_sfx(teacher)}.jsonl"],
                    outputs=[WORK / f"teacher_gate_report{_sfx(teacher)}.json"],
                    config={"thresholds": GATE, "pipeline": "v4"}, metrics=rep)
    return rep["passed"]


# ---------------------------------------------------------------- build
_SRC = re.compile(r"https?://\S+")


def _grounded(answer, quotes):
    ctx = set(tokens("\n".join(q["text"] for q in quotes)))
    toks = tokens(_SRC.sub(" ", answer))
    return sum(t in ctx for t in toks) / len(toks) if toks else 0.0


def build(teacher: str = "mistral-small-3.2"):
    from .v4 import EXTRACT_SYSTEM, ANSWER_SYSTEM, answer_messages, extract_messages
    rep = json.loads((WORK / f"teacher_gate_report{_sfx(teacher)}.json").read_text())
    if not rep["passed"]:
        sys.exit("teacher gate failed: its rows are not used")
    items_ = {r["item_id"]: r for r in _jsonl(WORK / "pipeline_items.jsonl")}
    ev_urls = {u for r in _eval_rows() for u in r["source_urls"]}
    drops, ext_rows, ans_rows = Counter(), [], []
    for r in _jsonl(WORK / f"teacher_pipeline_raw{_sfx(teacher)}.jsonl"):
        it = items_[r["item_id"]]
        assert not {h["url"] for h in it["hits"]} & ev_urls, "eval page in a training context"
        base = {"item_id": it["item_id"], "kind": it["kind"], "split": it["split"], "program": it["program"],
                "question": it["question"], "source_url": it.get("url")}
        if it["kind"] in ("answer", "situation"):
            if r["status"] != "answered":
                drops[f"{it['kind']}:teacher_{r['status']}"] += 1
                continue
            if _grounded(r["answer_raw"], r["quotes"]) < GROUNDED_MIN:
                drops[f"{it['kind']}:answer_not_grounded"] += 1
                continue
            idx = {h["url"]: i for i, h in enumerate(it["hits"], 1)}
            target = {"quotes": [{"source": idx.get(q["url"], 1), "text": q["text"]} for q in r["quotes"]]}
            m = extract_messages(it["question"], it["hits"])
            ext_rows.append({**base, "task": "extract", "instruction": EXTRACT_SYSTEM, "input": m[1]["content"],
                             "output": json.dumps(target, ensure_ascii=False)})
            m2 = answer_messages(it["question"], r["quotes"])
            ans_rows.append({**base, "task": "answer", "instruction": ANSWER_SYSTEM, "input": m2[1]["content"],
                             "output": r["answer_raw"].strip()})
        elif it["kind"] == "no_quote":
            if r["status"] != "refused_no_quotes":
                drops["no_quote:teacher_found_quotes"] += 1  # the answer was still reachable: not a refusal case
                continue
            m = extract_messages(it["question"], it["hits"])
            ext_rows.append({**base, "task": "extract", "instruction": EXTRACT_SYSTEM, "input": m[1]["content"],
                             "output": json.dumps({"quotes": []})})
        elif it["kind"] == "policy":
            if r["status"] != "refused_policy":
                drops[f"policy:teacher_{r['status']}"] += 1
                continue
            m = extract_messages(it["question"], it["hits"])
            ext_rows.append({**base, "task": "extract", "instruction": EXTRACT_SYSTEM, "input": m[1]["content"],
                             "output": json.dumps({"refuse": True})})
    # balance extract refusals to ~REFUSAL_SHARE, keeping every policy refusal
    rng = random.Random(SEED)
    pos = [r for r in ext_rows if r["kind"] in ("answer", "situation")]
    pol = [r for r in ext_rows if r["kind"] == "policy"]
    noq = [r for r in ext_rows if r["kind"] == "no_quote"]
    want_noq = max(0, round(len(pos) * REFUSAL_SHARE / (1 - REFUSAL_SHARE)) - len(pol))
    rng.shuffle(noq)
    ext_rows = pos + pol + noq[:want_noq]
    rows = ext_rows + ans_rows
    rng.shuffle(rows)
    train = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "valid"]
    _write(SFT / "train.jsonl", train)
    _write(SFT / "valid.jsonl", valid)
    card = _card(train, valid, drops)
    (SFT / "DATA_CARD.md").write_text(card["md"])
    print(card["md"].split("## 20 random")[0])
    tracking.record("sft_build:v4", inputs=[WORK / f"teacher_pipeline_raw{_sfx(teacher)}.jsonl", WORK / "pipeline_items.jsonl"],
                    outputs=[SFT / "train.jsonl", SFT / "valid.jsonl", SFT / "DATA_CARD.md"],
                    config={"teacher": teacher, "refusal_share": REFUSAL_SHARE, "grounded_min": GROUNDED_MIN, "seed": SEED},
                    metrics=card["metrics"])


def _card(train, valid, drops):
    ev = _eval_rows()
    ev_urls = {u for r in ev for u in r["source_urls"]}
    ev_q = {norm(r["question"]) for r in ev}
    allr = train + valid
    ctx_urls = lambda r: set(re.findall(r"https?://\S+", r["input"])) | {r["source_url"]} - {None}
    checks = {
        "no_eval_question_in_any_row": not any(norm(r["question"]) in ev_q for r in allr),
        "no_eval_page_as_source": not any(r["source_url"] in ev_urls for r in allr if r["source_url"]),
        "valid_and_train_share_no_page": not ({r["source_url"] for r in train if r["source_url"]} & {r["source_url"] for r in valid if r["source_url"]}),
        "extract_targets_are_valid_json": all(_parse_json(r["output"]) is not None for r in allr if r["task"] == "extract"),
        "refusal_share_of_extract_ge_0_20": sum(r["kind"] in ("no_quote", "policy") for r in train if r["task"] == "extract")
        / max(1, sum(r["task"] == "extract" for r in train)) >= 0.20,
    }
    c = lambda rs: dict(Counter(f"{r['task']}:{r['kind']}" for r in rs))
    m = {"train": len(train), "valid": len(valid), "train_kinds": c(train), "valid_kinds": c(valid), "drops": dict(drops),
         "checks": checks, "all_checks_pass": all(checks.values())}
    md = ["# SFT data card (Visa Buddy v4: Qwen3-4B, extract + answer)", "",
          f"- train rows: **{len(train)}** {c(train)}", f"- valid rows: **{len(valid)}** {c(valid)}",
          f"- dropped: {dict(drops)}", "", "## Checks", ""] + [f"- [{'x' if v else ' '}] {k}" for k, v in checks.items()] + [
          "", "## 20 random train rows", ""]
    for r in random.Random(SEED + 1).sample(train, min(20, len(train))):
        md += [f"### {r['task']} / {r['kind']} ({r['program']})", f"**Q:** {r['question']}", "", f"**Target:** {r['output'][:600]}", ""]
    return {"metrics": m, "md": "\n".join(md) + "\n"}


if __name__ == "__main__":
    cmd, arg = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    if cmd in ("gate", "build"):
        {"gate": gate, "build": build}[cmd](arg or "mistral-small-3.2")
    else:
        {"select": select, "items": items}[cmd]()
