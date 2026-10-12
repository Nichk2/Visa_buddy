from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "visa_buddy_data"
CORPUS_DIR = DATA / "corpus"
INDEX_DIR = DATA / "index"
EVAL_DIR = ROOT / "eval"
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"

GOLD_V1 = EVAL_DIR / "gold_v1.jsonl"

ALLOWED_HOST = "www.canada.ca"
START_URL = "https://www.canada.ca/en/immigration-refugees-citizenship.html"
IRCC_PREFIX = "/en/immigration-refugees-citizenship/services/"
USER_AGENT = "VisaBuddyResearchBot/0.2 (non-commercial portfolio project; polite crawl)"
CRAWL_DELAY_S = 1.0

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 200
TOP_K = 5

PROGRAMS = [
    "express_entry", "study_permit", "work_permit", "pgwp",
    "visitor", "family", "citizenship", "out_of_scope",
]

# Path prefix -> program tag. Longest match wins, so PGWP is tagged before
# the broader study-canada prefix.
PROGRAM_PREFIXES = {
    "immigrate-canada/express-entry": "express_entry",
    "study-canada/work/after-graduation": "pgwp",
    "study-canada": "study_permit",
    "work-canada": "work_permit",
    "visit-canada": "visitor",
    "immigrate-canada/family-sponsorship": "family",
    "canadian-citizenship": "citizenship",
}

# Deliberately not indexed, so must_refuse questions about them have nothing to hit.
EXCLUDED_PREFIXES = [
    "immigrate-canada/provincial-nominees",
    "refugees",
    "immigrate-canada/atlantic-immigration",
    "immigrate-canada/caregivers",
    "immigrate-canada/start-visa",
    "immigrate-canada/self-employed",
    "immigrate-canada/rural-franco",
    "immigrate-canada/rural-community",
]

# Official landing page used in refusals.
OFFICIAL_INDEX_URL = START_URL

# Hub pages the crawl starts from (besides START_URL).
SEED_PATHS = [
    "immigrate-canada/express-entry.html",
    "study-canada/study-permit.html",
    "study-canada/work.html",
    "study-canada/work/after-graduation.html",
    "work-canada.html",
    "visit-canada.html",
    "visit-canada/extend-stay.html",
    "immigrate-canada/family-sponsorship.html",
    "canadian-citizenship.html",
    "canadian-citizenship/adult-minor.html",
]
# Pages outside the program prefixes that are indexed under an explicit tag.
# Guide 5256 is the live page that visa_buddy_data/raw_pdfs/Visitor visa guide.pdf
# was printed from (PDF metadata, 2025-11-25); the PDF is not indexed separately.
EXTRA_URLS = {
    "https://www.canada.ca/en/immigration-refugees-citizenship/services/application/application-forms-guides/guide-5256-applying-visitor-visa-temporary-resident-visa.html": "visitor",
}
MAX_PAGES = 700
MAX_PAGES_PER_PROGRAM = 130
MAX_DEPTH = 4
MIN_DOC_CHARS = 200
