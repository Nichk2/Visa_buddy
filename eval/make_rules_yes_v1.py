"""Build eval/rules_yes_v1.jsonl: python eval/make_rules_yes_v1.py

Companion to rules_v1, whose correct answers are 21 "no" to 2 "yes": a model that always
opened with "No" would score 0 wrong verdicts there. These questions (written by hand on
2026-10-16, before any v4 code) all have the verdict "yes", from pages never in a training
input (gold source pages), so the verdict check can't be passed by a one-sided model.
"""
import json
from pathlib import Path

B = "https://www.canada.ca/en/immigration-refugees-citizenship/services/"
PG_ELIG = B + "study-canada/work/after-graduation/eligibility.html"
SP_OFF = B + "study-canada/work/work-off-campus.html"
WP_RESTORE = B + "work-canada/restore.html"
VI_GUIDE = B + "application/application-forms-guides/guide-5256-applying-visitor-visa-temporary-resident-visa.html"
VI_ETA_ELIG = B + "visit-canada/eta/eligibility.html"
CI_BRING = B + "canadian-citizenship/become-canadian-citizen/citizenship-ceremony/bring.html"
CI_WHEN = B + "canadian-citizenship/become-canadian-citizen/citizenship-ceremony/when.html"
EE_FUNDS = B + "immigrate-canada/express-entry/documents/proof-funds.html"

rows = []


def q(id, program, question, facts, keys, urls):
    rows.append(dict(id=id, question=question, program=program, verdict="yes", expected_facts=facts,
                     fact_keys=keys, source_urls=urls, must_refuse=False, category="rule", recent_change=False))


q("y01", "pgwp", "Can I still get a PGWP if I studied part-time in my final semester?",
  ["You may study part-time during your final semester"], [["part-time during your final semester"]], [PG_ELIG])
q("y02", "pgwp", "I transferred between two designated learning institutions. Can I still get a PGWP if my studies added up to at least 8 months?",
  ["The length of your studies must be at least 8 months"], [["at least 8 months"]], [PG_ELIG])
q("y03", "pgwp", "Part of my program was an overseas component that earned credit from my Canadian school. Can I still be eligible for a PGWP?",
  ["eligible for a PGWP as long as you earn a credit from a Canadian school"], [["credit from a Canadian school"]], [PG_ELIG])
q("y04", "study_permit", "While I study in Canada, can I work remotely for an employer outside Canada?",
  ["You can work remotely for an employer outside Canada", "doesn't count towards your 24 hours per week"],
  [["work remotely for an employer outside Canada"], ["doesn't count towards your 24 hours"]], [SP_OFF])
q("y05", "study_permit", "I'm in my last semester and only need a part-time course load. Can I still work off campus?",
  ["you're in the last semester of your study program and you don't need a full course load"],
  [["last semester"]], [SP_OFF])
q("y06", "work_permit", "Can I pay my restoration fees with a credit card that isn't in my name?",
  ["The card you use doesn't have to be in your name"], [["doesn't have to be in your name"]], [WP_RESTORE])
q("y07", "visitor", "Can my family submit our visitor visa applications together and use one payment receipt?",
  ["submit your applications together ... and use one payment receipt"], [["one payment receipt"]], [VI_GUIDE])
q("y08", "visitor", "I'm a US citizen. Can I fly to Canada without an eTA?",
  ["U.S. citizens are exempt from the eTA requirement"], [["U.S. citizens are exempt"]], [VI_ETA_ELIG])
q("y09", "citizenship", "Can I bring a holy book of my choice to swear the oath at my citizenship ceremony?",
  ["a holy book of your choice, if you want to use it to swear the oath of citizenship"], [["holy book"]], [CI_BRING])
q("y10", "citizenship", "My citizenship ceremony is virtual. Can my family watch it in the same room as me?",
  ["Your guests can be in the same physical room with you to watch the ceremony"], [["same physical room"]], [CI_WHEN])
q("y11", "express_entry", "My spouse is coming to Canada with me. Can we count money in our joint account as proof of funds for Express Entry?",
  ["If your spouse is coming with you, you can count money you have together in a joint account"],
  [["joint account"]], [EE_FUNDS])

if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("rules_yes_v1.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(len(rows), "rows ->", out)
