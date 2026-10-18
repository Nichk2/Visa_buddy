from pipeline.prompt import REFUSAL_OUTPUT
from pipeline.v4 import run_batch, verify_answer, verify_quotes

U = "https://www.canada.ca/x/pgwp.html"
HITS = [{"url": U, "title": "PGWP: Who can apply",
         "text": "You must complete at least 50% of your program within Canada (online or in class).\n"
                 "This temporary policy was in place until August 31, 2022."}]


def test_quote_must_exist_in_sources():
    ok = verify_quotes([{"source": 1, "text": "You must complete at least 50% of your program within Canada (online or in class)."}], HITS)
    assert ok and ok[0]["url"] == U and ok[0]["text"].startswith("You must complete")
    near = verify_quotes([{"source": 1, "text": "you must complete at least 50% of your program within canada"}], HITS)
    assert near and near[0]["text"].endswith("(online or in class).")  # shown as the real sentence
    assert verify_quotes([{"source": 1, "text": "You can study 100% online from abroad and still get a PGWP."}], HITS) == []


def test_answer_checks():
    q = verify_quotes([{"source": 1, "text": "You must complete at least 50% of your program within Canada (online or in class)."}], HITS)
    assert verify_answer("No. You must complete at least 50% of your program within Canada.", q) == []
    assert verify_answer("No. You must complete at least 60% of it in Canada.", q)  # number not in quotes
    assert verify_answer("You don't have to complete at least 50% of your program within Canada.", q)  # flipped


def scripted(responses):
    calls = iter(responses)
    return lambda convs, max_tokens: [next(calls) for _ in convs]


def test_run_batch_paths():
    good_quote = '{"quotes": [{"source": 1, "text": "You must complete at least 50% of your program within Canada (online or in class)."}]}'
    chat = scripted([good_quote, '{"quotes": []}', '{"refuse": true}', '{"quotes": [{"source": 1, "text": "Invented rule that is not on the page at all."}]}', good_quote,
                     "No. You must complete at least 50% of your program within Canada.",
                     "Yes, you don't have to complete at least 50% of your program within Canada."])
    r = run_batch(["q1", "q2", "q3", "q4", "q5"], [HITS] * 5, chat)
    assert [x["status"] for x in r] == ["answered", "refused_no_quotes", "refused_policy", "refused_no_quotes", "quotes_only"]
    assert r[0]["response"].startswith("No.") and "Official wording:" in r[0]["response"] and U in r[0]["response"]
    assert r[1]["response"] == r[2]["response"] == REFUSAL_OUTPUT
    assert "Yes" not in r[4]["response"] and "official sources say" in r[4]["response"]  # wrong answer never shown


def test_multiline_and_string_quotes():
    hits = [{"url": U, "title": "Express Entry", "text": "There are 3 immigration programs managed through Express Entry:\n"
             "- Canadian Experience Class\n- Federal Skilled Worker Program\n- Federal Skilled Trades Program\nOther text here that is long enough."}]
    q = "There are 3 immigration programs managed through Express Entry: - Canadian Experience Class - Federal Skilled Worker Program - Federal Skilled Trades Program"
    for raw in ([{"source": 1, "text": q}], [q]):  # object form and plain-string form (Phi-4-mini)
        got = verify_quotes(raw, hits)
        assert got and "Federal Skilled Trades Program" in got[0]["text"] and "Other text" not in got[0]["text"]


def test_expired_quotes():
    from datetime import date
    from pipeline.v4 import is_expired
    today = date(2026, 10, 17)
    assert is_expired("You can complete up to 100% of your studies online from outside Canada between March 2020 and "
                      "August 31, 2022, because of a temporary COVID-19 policy.", today)
    assert is_expired("This temporary policy was in place until August 31, 2022.", today)
    assert not is_expired("If you submitted your PGWP application before November 1, 2024, you're exempt from the new requirements.", today)
    assert not is_expired("We will not accept new forms until further notice.", today)
    assert not is_expired("The temporary measures have been extended until November 27, 2026.", today)  # still in force


def test_expired_only_quotes_get_one_retry_without_them():
    from datetime import date
    hits = [{"url": U, "title": "PGWP", "text": "You can complete up to 100% of your studies online from outside Canada between March 2020 "
             "and August 31, 2022, because of a temporary COVID-19 policy.\nYou must complete at least 50% of your program within Canada."}]
    old = '{"quotes": [{"source": 1, "text": "You can complete up to 100% of your studies online from outside Canada between March 2020 and August 31, 2022, because of a temporary COVID-19 policy."}]}'
    new = '{"quotes": [{"source": 1, "text": "You must complete at least 50% of your program within Canada."}]}'
    seen = []

    def chat(convs, max_tokens):
        seen.append(convs)
        return {1: [old], 2: [new], 3: ["No. You must complete at least 50% of your program within Canada."]}[len(seen)]
    r = run_batch(["Online from abroad, PGWP?"], [hits], chat, today=date(2026, 10, 17))[0]
    assert r["expired_dropped"] and r["status"] == "answered" and r["response"].startswith("No.")
    assert "August 31, 2022" not in seen[1][0][1]["content"]  # the retry never saw the expired line


def test_multiline_quote_keeps_line_breaks():
    hits = [{"url": U, "title": "Funds", "text": "- you can't use equity on real property as proof of settlement funds\n- you can't borrow this money from another person"}]
    q = verify_quotes(["you can't use equity on real property as proof of settlement funds you can't borrow this money from another person"], hits)
    assert q and "\n" in q[0]["text"]


def test_placeholder_quotes_dropped():
    hits = [{"url": U, "title": "Fees", "text": "Sponsor fee: [Error loading fee. Check the fee list ] for each application."}]
    assert verify_quotes(["Sponsor fee: [Error loading fee. Check the fee list ] for each application."], hits) == []


def test_verdict_conflict():
    from pipeline.v4 import verdict_conflict
    assert verdict_conflict("Yes, you do not have to show language test results because you applied before November 1, 2024.")
    assert verdict_conflict("No, you can apply online.")
    for ok in ["No. You can't use equity on real property.", "Yes, you must stop working.", "No. You must declare all family members.",
               "Yes. You can work remotely for an employer outside Canada.", "You must complete at least 50% in Canada.",
               "No. You can’t get a PGWP if you already had one.",  # curly apostrophe
               "No. You can only start working when you start your study program.",  # "only" restricts
               "Yes, you may study part-time in your final semester, and this does not affect eligibility."]:  # later clause
        assert not verdict_conflict(ok), ok


def test_section_level_expiry():
    from datetime import date
    from pipeline.v4 import _expired_pages, _in_expired_section
    # the dated sentence and the "this temporary policy" sentence are in different chunks of one page
    hits = [{"url": U, "title": "PGWP", "text": "You can complete up to 100% of your studies online from outside Canada between March 2020 and "
             "August 31, 2022, because of a temporary COVID-19 policy."},
            {"url": U, "title": "PGWP", "text": "You're eligible for this temporary policy if you meet all of the following criteria:\n"
             "You must complete at least 50% of your program within Canada."}]
    exp = _expired_pages(hits, date(2026, 10, 18))
    assert _in_expired_section("You're eligible for this temporary policy if you meet all of the following criteria:", U, exp)
    assert not _in_expired_section("You must complete at least 50% of your program within Canada.", U, exp)
