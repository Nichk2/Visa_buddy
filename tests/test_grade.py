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
