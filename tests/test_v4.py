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
