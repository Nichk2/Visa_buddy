"""Build eval/gold_v2.jsonl from the frozen gold_v1.jsonl.

    python eval/make_gold_v2.py

gold_v2 keeps every v1 question, program, expected fact and must_refuse flag unchanged.
It adds two things, recorded here so the change is reviewable:

1. fact_keys: for each expected fact, accepted spans that carry it. Rule applied to
   every fact (not only facts a model missed): the smallest span that states the fact,
   i.e. a number with its unit, a defined term, or the negation that makes it a rule
   ("can't sponsor"). Grader v2 canonicalizes money ("CAN$7" = "$7 CAD" = "$7") and
   contractions ("cannot" = "can't") before matching. Each fact must have at least one
   key on one of its source pages (checked by pipeline.gold_check).
2. extra_source_urls: other official pages that state all of a question's facts,
   accepted after reading them (see ACCEPTED_SOURCES).

Why: grader v1 matched facts close to the page's wording, so correct paraphrases
("$7 CAD" for "CAN$7", "no more than 90 days after" for "within 90 days") scored as
misses, and questions answered word-for-word on a second official page counted as
retrieval misses.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

FACT_KEYS = {
    "ee01": [["Canadian Experience Class"], ["Federal Skilled Worker"], ["Federal Skilled Trades"]],
    "ee02": [["1 year"], ["3 years"]],
    "ee03": [["CLB 7"]],
    "ee04": [["$15,263"]],
    "ee05": [["Canadian Experience Class"], ["letter explaining", "upload a letter"]],
    "ee06": [["can't use equity"], ["can't borrow"]],
    "ee07": [["studying full-time"], ["co-op"]],
    "ee08": [["job offer of full-time employment", "full-time job offer"], ["certificate of qualification"]],
    "sp01": [["$150"]],
    "sp02": [["$23,448"]],
    "sp03": [["$35,040"]],
    "sp04": [["24 hours"]],
    "sp05": [["unlimited hours"], ["scheduled break", "scheduled breaks", "break scheduled", "regular breaks"]],
    "sp06": [["ESL", "second language"], ["need to apply for a work permit", "need a work permit", "apply for a work permit"]],
    "sp07": [["DLI", "designated learning institution"], ["tuition"], ["leave Canada when your study permit expires"]],
    "sp08": [["6 months"]],
    "pg01": [["180 days"]],
    "pg02": [["8 months"], ["900 hours"]],
    "pg03": [["CLB 7", "NCLC 7"], ["4 language areas", "all four language areas"]],
    "pg04": [["CLB 5", "NCLC 5"], ["4 language areas", "all four language areas"]],
    "pg05": [["already had one", "already received a PGWP", "already had a PGWP", "only get one PGWP"]],
    "pg06": [["3-year", "3 years"], ["8 months"], ["certificate or diploma"]],
    "pg07": [["$255"]],
    "pg08": [["90 days"], ["restore your status"]],
    "wp01": [["any employer"], ["conditions of the permit", "conditions on your permit", "conditions listed"],
             ["employer, location and occupation", "specific employer"]],
    "wp02": [["don't need a job offer"]],
    "wp03": [["labour market impact assessment"], ["hire a foreign worker"], ["most jobs need an LMIA"]],
    "wp04": [["business visitors"], ["exempt", "don't need a work permit"]],
    "wp05": [["90 days"]],
    "wp06": [["change employers at any time", "can change employers"]],
    "wp07": [["new job offer"], ["stop working"]],
    "vi01": [["6 months"]],
    "vi02": [["visitor record"], ["before your current status expires", "before your status expires"]],
    "vi03": [["30 days"]],
    "vi04": [["$7"]],
    "vi05": [["within minutes"], ["72 hours"]],
    "vi06": [["5 years"], ["10 years"]],
    "vi07": [["$100"]],
    "vi08": [["exempt from the eTA requirement", "don't need an eTA"]],
    "fa01": [["18 years"], ["Canadian citizen"], ["live in Canada"], ["undertaking"]],
    "fa02": [["isn't an income requirement", "no income requirement"], ["dependent children of their own", "1 or more dependent children"]],
    "fa03": [["can't sponsor"]],
    "fa04": [["less than 5 years", "5 years"]],
    "fa05": [["not currently accepting", "not accepting"], ["paused"]],
    "fa06": [["super visa"], ["5 years"], ["2-year extension", "2-year extensions"]],
    "fa07": [["social assistance"]],
    "ci01": [["1,095 days"], ["5-year", "5 years"]],
    "ci02": [["0.5 days", "half a day"], ["365 days"]],
    "ci03": [["CLB 4"], ["18 to 54", "18 and 54"]],
    "ci04": [["18 and 54", "18 to 54"]],
    "ci05": [["3 years"]],
    "ci06": [["rights and responsibilities"], ["geography"]],
    "ci07": [["expired PR card"]],
    "ci08": [["oath of citizenship"]],
}

# Candidates: pages in the question's program that contain every expected fact under both
# v1 matching and v2 keys (86 found). Accepted only when the page answers this question for
# the same audience; rejected when the facts appear in another context (e.g. the Francophone
# student pilot's fund tables, parents' sponsorship rules for a spouse question, "90 days"
# for LMIA documents vs restoring status). All decisions: eval/gold_v2_source_review.jsonl.
B = "https://www.canada.ca/en/immigration-refugees-citizenship/services/"
ACCEPTED_SOURCES: dict[str, list[str]] = {
    "ee01": [
        B + "immigrate-canada/express-entry.html",
    ],
    "ee03": [
        B + "immigrate-canada/express-entry/documents/language-test.html",
    ],
    "ee08": [
        B + "immigrate-canada/express-entry/who-can-apply/federal-skilled-trades.html",
    ],
    "pg07": [
        B + "study-canada/work/after-graduation.html",
    ],
    "wp02": [
        B + "work-canada/need-permit.html",
        B + "work-canada/permit-poe/eligibility.html",
        B + "work-canada/special-instructions.html",
    ],
    "wp07": [
        B + "work-canada/extend/change-jobs-employers/lose-job.html",
    ],
    "vi01": [
        B + "visit-canada/about-visitor-visa.html",
        B + "visit-canada/prepare-arrival.html",
        B + "visit-canada/eta/after-apply-next-steps.html",
        B + "visit-canada/eta/about.html",
    ],
    "vi04": [
        B + "visit-canada/eta/about.html",
    ],
    "vi05": [
        B + "visit-canada/eta/about.html",
        B + "visit-canada/eta/after-apply-next-steps.html",
    ],
    "vi06": [
        B + "application/application-forms-guides/guide-5256-applying-visitor-visa-temporary-resident-visa.html",
        B + "visit-canada/apply-visitor-visa.html",
        B + "visit-canada/check-visa-eta.html",
        B + "visit-canada/entry-requirements-country.html",
    ],
    "vi08": [
        B + "visit-canada/eta/eligibility.html",
        B + "visit-canada/eta/facts.html",
        B + "visit-canada/eta/important-reminders-apply-eta.html",
    ],
    "fa05": [
        B + "immigrate-canada/family-sponsorship/sponsor-parents-grandparents/apply.html",
    ],
    "fa06": [
        B + "immigrate-canada/family-sponsorship/sponsor-parents-grandparents/selected.html",
    ],
    "ci03": [
        B + "canadian-citizenship/adult-minor/how/adult-forms-documents.html",
    ],
    "ci04": [
        B + "canadian-citizenship/adult-minor/after.html",
        B + "canadian-citizenship/become-canadian-citizen/citizenship-ceremony/when.html",
        B + "canadian-citizenship/become-canadian-citizen/eligibility/waiver/who.html",
        B + "canadian-citizenship/test/how-it-works.html",
    ],
    "ci07": [
        B + "canadian-citizenship/adult-minor/how/adult-forms-documents.html",
    ],
    "ci08": [
        B + "canadian-citizenship/become-canadian-citizen/citizenship-ceremony/expect.html",
        B + "canadian-citizenship/become-canadian-citizen/citizenship-ceremony/after.html",
        B + "canadian-citizenship/become-canadian-citizen/citizenship-ceremony/who.html",
        B + "canadian-citizenship/become-canadian-citizen/citizenship-ceremony/bring.html",
        B + "canadian-citizenship/become-canadian-citizen/after-apply-next-steps/status.html",
        B + "canadian-citizenship/take-part-citizenship.html",
        B + "canadian-citizenship/take-part-citizenship/citizenship-week.html",
        B + "canadian-citizenship/take-part-citizenship/host-citizenship-ceremony.html",
        B + "canadian-citizenship/take-part-citizenship/host-citizenship-ceremony/about.html",
        B + "canadian-citizenship/take-part-citizenship/host-citizenship-ceremony/who-you-work-with.html",
    ],
}


def main():
    rows = [json.loads(l) for l in (HERE / "gold_v1.jsonl").read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        r = dict(r)
        if not r["must_refuse"]:
            keys = FACT_KEYS[r["id"]]
            assert len(keys) == len(r["expected_facts"]), r["id"]
            r["fact_keys"] = keys
            extra = [u for u in ACCEPTED_SOURCES.get(r["id"], []) if u not in r["source_urls"]]
            r["source_urls"] = r["source_urls"] + extra
            r["extra_source_urls"] = extra
        out.append(r)
    (HERE / "gold_v2.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"wrote eval/gold_v2.jsonl: {len(out)} rows, "
          f"{sum(len(r.get('extra_source_urls', [])) for r in out)} extra source URLs")


if __name__ == "__main__":
    main()
