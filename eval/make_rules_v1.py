"""Build eval/rules_v1.jsonl: python eval/make_rules_v1.py

Held-out rule questions, written by hand on 2026-10-16 before any v4 work, from pages
that have never been in a training input (they are gold source pages, held out of every
SFT version). Each question has a yes/no verdict, because the failure we're targeting is
a model stating the opposite rule (v2/v3: "No, you can't get a 3-year PGWP").
recent_change marks rules introduced or changed in 2024-2025, where a model trained
earlier is most likely to answer from stale memory.
"""
import json
from pathlib import Path

B = "https://www.canada.ca/en/immigration-refugees-citizenship/services/"
PG_ABOUT = B + "study-canada/work/after-graduation/about.html"
PG_ELIG = B + "study-canada/work/after-graduation/eligibility.html"
PG_APPLY = B + "study-canada/work/after-graduation/apply.html"
SP_OFF = B + "study-canada/work/work-off-campus.html"
SP_FIN = B + "study-canada/study-permit/get-documents/financial-support.html"
EE_LANG = B + "immigrate-canada/express-entry/documents/language-test.html"
EE_CEC = B + "immigrate-canada/express-entry/who-can-apply/canadian-experience-class.html"
EE_FUNDS = B + "immigrate-canada/express-entry/documents/proof-funds.html"
CI_WHO = B + "canadian-citizenship/adult-minor/who.html"
CI_EXPECT = B + "canadian-citizenship/become-canadian-citizen/citizenship-ceremony/expect.html"
CI_AFTER = B + "canadian-citizenship/become-canadian-citizen/citizenship-ceremony/after.html"
VI_GUIDE = B + "application/application-forms-guides/guide-5256-applying-visitor-visa-temporary-resident-visa.html"
VI_ETA_FACTS = B + "visit-canada/eta/facts.html"
WP_RESTORE = B + "work-canada/restore.html"
FA_ELIG = B + "immigrate-canada/family-sponsorship/spouse-partner-children/eligibility.html"

rows = []


def q(id, program, question, verdict, facts, keys, urls, recent=False):
    rows.append(dict(id=id, question=question, program=program, verdict=verdict, expected_facts=facts,
                     fact_keys=keys, source_urls=urls, must_refuse=False, category="rule",
                     recent_change=recent))


q("r01", "pgwp", "I submitted my PGWP application in October 2024. Do I have to show language test results?", "no",
  ["submitted your PGWP application before November 1, 2024 ... exempt from the new requirements"],
  [["before November 1, 2024"]], [PG_ABOUT], recent=True)
q("r02", "pgwp", "I started a program in June 2024 at a private college that delivers a public college's program in the same province. Can I get a PGWP?", "no",
  ["began one of these programs after May 15, 2024, you're not eligible"], [["May 15, 2024"]], [PG_ELIG], recent=True)
q("r03", "pgwp", "Can I take my whole program online from outside Canada and still be eligible for a PGWP?", "no",
  ["complete at least 50% of your program within Canada"], [["50%"]], [PG_ELIG], recent=True)
q("r04", "pgwp", "I studied 4 months at a school that isn't a DLI and then 6 months at a DLI. Can I get a PGWP?", "no",
  ["Only the time you spent studying at the DLI will be counted", "at least 8 months at the DLI"],
  [["only the time you spent studying at the DLI"], ["8 months"]], [PG_ELIG])
q("r05", "pgwp", "My study permit had already expired when I applied for my PGWP. Can I work while I wait for the decision?", "no",
  ["Your study permit must be valid (not expired) at the time you submit your application"], [["must be valid"]], [PG_APPLY])
q("r06", "pgwp", "My passport expires before my PGWP would. Once I get a new passport, can I extend my PGWP online?", "no",
  ["you must apply on paper"], [["on paper"]], [PG_ABOUT])
q("r07", "study_permit", "I'm on an authorized leave from my studies. Can I keep working off campus?", "no",
  ["you can't work off campus", "only return to work once you resume your studies"],
  [["can't work off campus"], ["resume your studies"]], [SP_OFF])
q("r08", "study_permit", "Can I start working off campus a few weeks before my study program begins?", "no",
  ["You can't work before your studies begin"], [["can't work before your studies begin", "only start working"]], [SP_OFF])
q("r09", "study_permit", "My scholarship covers only my tuition. Is that enough proof of funds for my study permit?", "no",
  ["provide other documents proving you have sufficient funds to cover living expenses and transportation costs"],
  [["living expenses"]], [SP_FIN])
q("r10", "study_permit", "My program is 3 years long. Do I only need to show money for the first year of my studies?", "no",
  ["tell us how you plan to pay for the full duration of your studies"], [["full duration"]], [SP_FIN], recent=True)
q("r11", "express_entry", "Can I use English test results that are 3 years old for my Express Entry application?", "no",
  ["Your test results must be less than 2 years old"], [["less than 2 years old"]], [EE_LANG])
q("r12", "express_entry", "I live in Canada and work remotely for a company in another country. Does that count for the Canadian Experience Class?", "no",
  ["physically in Canada and working for a Canadian employer"], [["Canadian employer"]], [EE_CEC])
q("r14", "express_entry", "Can I count money in my spouse's own bank account as proof of funds for Express Entry?", "yes",
  ["You may be able to count money in an account under your spouse's name only, but you must prove you have access"],
  [["prove you have access"]], [EE_FUNDS])
q("r15", "citizenship", "I'm 60 years old. Do I have to take the Canadian citizenship test?", "no",
  ["between 18 and 54 years old ... must take the citizenship test"], [["18 and 54"]], [CI_WHO])
q("r16", "citizenship", "Can I wear perfume to my citizenship ceremony?", "no",
  ["You cannot wear scented products"], [["scented products"]], [CI_EXPECT])
q("r17", "citizenship", "Can I sign my oath or affirmation form the day before my citizenship ceremony?", "no",
  ["sign this form on the day you take the oath of citizenship, not the day before or after"],
  [["not the day before"]], [CI_EXPECT])
q("r18", "citizenship", "After I become a citizen, can I travel abroad and come back to Canada using my old PR card?", "no",
  ["you must have a Canadian passport to travel abroad and return to Canada"], [["Canadian passport"]], [CI_AFTER])
q("r19", "visitor", "Can I get a Canadian visitor visa when I land at the airport in Canada?", "no",
  ["You cannot obtain one upon arrival in Canada"], [["upon arrival"]], [VI_GUIDE])
q("r20", "visitor", "I'm from a visa-exempt country. Do I need an eTA if I drive into Canada from the US?", "no",
  ["do not need an eTA when arriving by car"], [["by car"]], [VI_ETA_FACTS])
q("r21", "visitor", "I got a new passport. Is the eTA linked to my old passport still valid for my next flight?", "no",
  ["If you get a new passport, you need to get a new eTA"], [["new eTA"]], [VI_ETA_FACTS])
q("r22", "work_permit", "My family members lost their status too. Can we restore it with one application and one fee?", "no",
  ["they must each apply separately", "There is no group or family rate"],
  [["apply separately"], ["no group or family rate"]], [WP_RESTORE])
q("r23", "work_permit", "My work permit and status expired. Can I apply straight for a different kind of permit without restoring first?", "no",
  ["you must restore your initial permit and status first"], [["restore your initial permit"]], [WP_RESTORE])
q("r24", "family", "I hold Canadian citizenship but I work in Dubai right now. Is it possible for me to sponsor my wife for permanent residence?", "yes",
  ["you must show that you plan to live in Canada when the people you're sponsoring become permanent residents"],
  [["plan to live in Canada"]], [FA_ELIG])

if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("rules_v1.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(len(rows), "rows ->", out)
