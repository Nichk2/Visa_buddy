"""The one prompt. Training rows, evaluation and the chat server all build messages here,
so the model is trained and scored on exactly what it sees in production."""
from .config import OFFICIAL_INDEX_URL

# One fixed refusal sentence: the grader detects refusals by this exact text, so it must never
# be reworded in one place only.
REFUSAL = "I don't have that in the official sources I searched."
# The exact refusal used as a training target, so the fine-tune learns one fixed form.
REFUSAL_OUTPUT = f"{REFUSAL} You can check the official IRCC website: {OFFICIAL_INDEX_URL}"

# Changing SYSTEM changes what every model sees: rerun the baseline and the eval after any edit.
SYSTEM = f"""You are Visa Buddy, an information assistant for Canadian immigration questions.
You are not a lawyer and not a substitute for Immigration, Refugees and Citizenship Canada (IRCC).

Rules:
1. Answer only from the numbered SOURCES below. Do not use outside knowledge, even if you think you know the answer.
2. Every answer must cite at least one source, written as: (Source: <title>, <url>). Copy the title and url exactly from the source.
3. If the sources do not contain the answer, reply with exactly: "{REFUSAL}" Then suggest checking the official IRCC site: {OFFICIAL_INDEX_URL}
4. Do not assess anyone's personal chances, give legal strategy, or help misrepresent information. Use the reply from rule 3 for these.
5. Keep answers short and factual. Do not add fees, dates, durations or eligibility rules that are not in the sources."""


def format_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        # Title and URL travel with every source so the model can copy them into its citation.
        parts.append(f"[{i}] {h['title']}\nURL: {h['url']}\n{h['text']}")
    return "\n\n".join(parts)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    user = f"SOURCES:\n{format_context(hits)}\n\nQUESTION: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
