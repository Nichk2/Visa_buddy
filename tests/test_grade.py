from pipeline.grade import grade
from pipeline.prompt import REFUSAL

U = "https://www.canada.ca/x/proof-funds.html"
HITS = [{"url": U, "title": "Proof of funds", "text": "A single applicant needs $15,263. Updated July 7, 2025."}]
GOLD = {"expected_facts": ["$15,263"], "must_refuse": False}
REF = {"expected_facts": [], "must_refuse": True}


def test_good_answer():
    g = grade(f"You need $15,263 (Source: Proof of funds, {U}).", HITS, GOLD)
    assert g["citation_hit"] and g["fact_coverage"] == 1 and not g["unsupported"] and g["refusal_correct"]


def test_invented_number_is_unsupported():
    g = grade(f"You need $15,263 plus a $500 fee within 30 days (Source: {U}).", HITS, GOLD)
    assert g["unsupported"] and "$500" in g["unsupported_claims"]


def test_fact_must_be_in_chunks_not_just_answer():
    hits = [{"url": U, "title": "t", "text": "Nothing about money here."}]
    g = grade(f"You need $15,263 ({U}).", hits, GOLD)
    assert g["fact_coverage"] == 0


def test_url_not_retrieved_is_not_a_citation_hit():
    g = grade("You need $15,263 (Source: https://www.canada.ca/other.html).", HITS, GOLD)
    assert not g["citation_hit"] and g["unsupported"]


def test_refusal_scoring():
    assert grade(REFUSAL + " Check https://www.canada.ca/en/immigration-refugees-citizenship.html", HITS, REF)["refusal_correct"]
    assert not grade(REFUSAL, HITS, GOLD)["refusal_correct"]
    assert not grade(f"$15,263 ({U})", HITS, REF)["refusal_correct"]


def test_source_markers_and_dates_ok():
    g = grade(f"Per [1], updated July 7, 2025, it's $15,263 ({U}).", HITS, GOLD)
    assert not g["unsupported"], g["unsupported_claims"]


def test_digits_inside_cited_url_ignored():
    u = "https://www.canada.ca/guide-5256-visitor.html"
    hits = [{"url": u, "title": "Guide 5256", "text": "The fee is $100."}]
    g = grade(f"The fee is $100 (Source: Guide, {u}).", hits, {"expected_facts": ["$100"], "must_refuse": False})
    assert not g["unsupported"], g["unsupported_claims"]


def test_number_needs_its_unit():
    hits = [{"url": U, "title": "t", "text": "You can stay 10 days. A super visa lasts 5 years at a time."}]
    gold = {"expected_facts": [], "must_refuse": True}
    assert "10 points" in grade("It gives 10 points.", hits, gold)["unsupported_claims"]
    assert not grade("Stay up to 10 days or 5 years at a time.", hits, gold)["unsupported"]
    assert grade("It costs $10.", hits, gold)["unsupported"]


def test_v2_money_and_contractions():
    from pipeline.text import key_in
    assert key_in("$7", "It costs $7 CAD.") and key_in("$7", "only costs CAN$7") and key_in("$100", "Fees: $CAN 100")
    assert key_in("can't sponsor", "you cannot sponsor") and not key_in("$7", "$70")
    assert not key_in("can change employers", "you can't change employers")


def test_v2_uses_keys_and_v1_unchanged():
    hits = [{"url": U, "title": "t", "text": "It only costs CAN$7."}]
    gold = {"expected_facts": ["CAN$7"], "fact_keys": [["$7"]], "must_refuse": False}
    assert grade(f"It is $7 CAD ({U}).", hits, gold)["fact_coverage"] == 1
    assert grade(f"It is $7 CAD ({U}).", hits, gold, version=1)["fact_coverage"] == 0


def test_v2_clb_level_forms():
    from pipeline.text import key_in
    assert key_in("CLB 7", "CLB level 7 or higher") and key_in("CLB 7", "Benchmarks (CLB) 7 in English")
    assert not key_in("CLB 7", "CLB 70")


def test_v3_polarity_catches_t04_contradiction():
    from pipeline.text import key_polarity_conflict
    src = "If you lose your status. You must stop working."
    assert key_polarity_conflict("stop working", "No, you don't have to stop working while you apply.", src)
    assert not key_polarity_conflict("stop working", "Yes, you must stop working.", src)


def test_v3_polarity_leading_no_and_negated_keys():
    from pipeline.text import key_polarity_conflict
    src = "If you're a permanent resident living outside Canada, you can't sponsor."
    assert not key_polarity_conflict("can't sponsor", "No, you cannot sponsor your spouse.", src)
    src2 = "You can't use equity on real property as proof of settlement funds."
    assert not key_polarity_conflict("can't use equity", "No. You can't use equity in your house.", src2)


def test_v3_polarity_clause_scope_and_double_negative():
    from pipeline.text import key_polarity_conflict
    src = "If you don't apply within 90 days, you must leave Canada."
    assert not key_polarity_conflict("leave Canada", "You'll need to leave Canada and reapply.", src)
    src2 = "You can't work without a work permit."
    assert not key_polarity_conflict("work permit", "You can't work in Canada without a work permit.", src2)


def test_v3_grade_withholds_credit_on_contradiction():
    hits = [{"url": U, "title": "Restore", "text": "You must stop working."}]
    gold = {"expected_facts": ["You must stop working"], "fact_keys": [["stop working"]], "must_refuse": False}
    bad = grade(f"No, you don't have to stop working ({U}).", hits, gold)
    assert bad["fact_coverage"] == 0 and bad["contradicts_source"]
    good = grade(f"Yes, you must stop working ({U}).", hits, gold)
    assert good["fact_coverage"] == 1 and not good["contradicts_source"]
    assert grade(f"No, you don't have to stop working ({U}).", hits, gold, version=2)["fact_coverage"] == 1  # v2 reproducible


def test_v3_numbers_in_titles_are_supported():
    hits = [{"url": U, "title": "Find your NOC - version 2016", "text": "Enter your job title."}]
    g = {"expected_facts": [], "must_refuse": True}
    assert not grade("Use the 2016 table.", hits, g)["unsupported"]
    assert grade("Use the 2016 table.", hits, g, version=2)["unsupported"]


def test_v3_false_alarm_patterns_from_precision_review():
    from pipeline.text import key_polarity_conflict as c
    # table rows flattened into one clause (ee03)
    assert not c("CLB 7", "The minimum is CLB 7 in all 4 abilities.",
                 "CLB 8\n5\nCLB 7\n4\nBelow CLB 7\nNot eligible to apply")
    # negation after the key (t05)
    assert not c("lose your student status", "You can lose your student status and may not be approved again.",
                 "You can lose your student status for doing this.")
    # negation in the main clause, key in the condition (ee05)
    assert not c("Canadian Experience Class", "You don't need proof of funds if you're applying under the Canadian Experience Class.",
                 "You don't need to show proof of funds if:\n- you're applying under the Canadian Experience Class")
    # heading line with "not" above the sentence that holds the key (fa06)
    assert not c("5 years", "A super visa lets them stay for 5 years at a time.",
                 "If you were not invited to apply\nA super visa lets them stay in Canada for 5 years at a time.")


def test_v3_real_contradictions_still_caught():
    from pipeline.text import key_polarity_conflict as c
    assert c("3-year", "You can't get a 3-year PGWP if your master's program was only 1 year.",
             "You can apply for a 3-year PGWP, even if your master's program was less than 2 years")
    assert c("stop working", "No, you don't have to stop working while you apply.", "You must stop working.")


def test_polarity_flips_for_training_rows():
    from pipeline.text import polarity_flips
    src = "If you lose your status, you must stop working.\nYou must apply within 90 days of losing your status."
    assert polarity_flips("No, you don't have to stop working if you lose your status.", src)
    assert not polarity_flips("Yes, you must stop working if you lose your status.", src)
    assert not polarity_flips("You must apply within 90 days of losing your status (Source: Restore, https://x.ca/a.html).", src)


def test_guard_keeps_answer_and_quotes_source():
    from pipeline.guard import NOTE, guard
    from pipeline.prompt import REFUSAL_OUTPUT
    hits = [{"url": U, "title": "PGWP", "text": "You can apply for a 3-year PGWP, even if your master's program was less than 2 years, as long as\n"
                                                 "- your program was at least 8 months\n- you meet all other eligibility requirements"}]
    ans = f"No, you can't apply for a 3-year PGWP if your master's program was less than 2 years ({U})."
    shown, flags = guard(ans, hits)
    assert flags and shown.startswith(ans) and NOTE in shown and "at least 8 months" in shown  # continuation lines kept
    assert guard(REFUSAL_OUTPUT, hits) == (REFUSAL_OUTPUT, [])  # refusals untouched
    ok = f"You can apply for a 3-year PGWP even if your master's program was less than 2 years ({U})."
    assert guard(ok, hits) == (ok, [])
