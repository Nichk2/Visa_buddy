"""Download and extract one canada.ca page. Used by both gold authoring and the crawler,
so the text a gold question was written from is the same text that gets indexed."""
import time
import urllib.robotparser
from datetime import datetime, timezone
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .config import ALLOWED_HOST, CRAWL_DELAY_S, IRCC_PREFIX, PROGRAM_PREFIXES, EXCLUDED_PREFIXES, USER_AGENT

_session = requests.Session()
_session.headers["User-Agent"] = USER_AGENT
_robots = urllib.robotparser.RobotFileParser(f"https://{ALLOWED_HOST}/robots.txt")
_robots_loaded = False
_last_fetch = 0.0


def normalize_url(url: str) -> str:
    # Drop #fragments and ?queries so one page has one URL; gold source_urls and the
    # train/held-out split compare URLs as strings.
    url, _ = urldefrag(url)
    p = urlparse(url)
    return f"https://{p.netloc}{p.path}"


def allowed(url: str) -> bool:
    # Respect robots.txt (canada.ca disallows search pages, menus and some PDFs).
    global _robots_loaded
    if urlparse(url).netloc != ALLOWED_HOST:
        return False
    if not _robots_loaded:
        _robots.read()
        _robots_loaded = True
    return _robots.can_fetch(USER_AGENT, url)


def program_for(url: str) -> str | None:
    """Program tag from the URL path, or None if the page is outside the indexed programs."""
    path = urlparse(url).path
    if not path.startswith(IRCC_PREFIX):
        return None
    rest = path[len(IRCC_PREFIX):]
    if any(rest.startswith(x) for x in EXCLUDED_PREFIXES):
        return None
    best = None  # longest matching prefix wins (PGWP pages live under study-canada)
    for prefix, prog in PROGRAM_PREFIXES.items():
        if rest.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, prog)
    return best[1] if best else None


def _clean_main(soup: BeautifulSoup):
    # Keep the page body, drop site chrome (language toggle, "page details", feedback form).
    # Note: ".provisional" is NOT removed: it wraps real content like the spouse-sponsorship
    # eligibility checklist, which an earlier version of this list silently deleted.
    main = soup.find("main") or soup.body
    for sel in ["nav", "script", "style", "noscript", "#wb-lng", ".pagedetails",
                ".gc-prtts", "section.gc-prtts", ".pagination", "#wb-dtmd", "form",
                "#gc-pft"]:
        for el in main.select(sel):
            el.decompose()
    return main


# Elements that can hold other text blocks. A list item that contains a nested list is
# read for its own lead-in text only ("prove you have enough money to pay for"), then its
# children are read separately; reading the whole thing duplicated or lost text.
_BLOCKS = ["p", "li", "ul", "ol", "table", "details", "div", "section"]


def _own_text(el) -> str:
    """Text of el excluding nested block children (those are visited separately)."""
    parts = []
    for c in el.children:
        if getattr(c, "name", None) in _BLOCKS:
            continue
        parts.append(c.get_text(" ", strip=True) if hasattr(c, "get_text") else str(c))
    return " ".join(" ".join(parts).split())


def _main_text(main) -> str:
    lines = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "h5", "summary", "p", "li", "td", "th", "dt", "dd"]):
        if el.find(_BLOCKS):  # container: keep its own lead-in text, children are visited
            t = _own_text(el)
        else:
            t = " ".join(el.get_text(" ", strip=True).split())
        if not t:
            continue
        if el.name in ("h1", "h2", "h3", "h4", "h5", "summary"):
            t = f"\n## {t}"  # "## " marks a section; the chunker splits on it and keeps the heading
        elif el.name == "li":
            t = f"- {t}"
        lines.append(t)
    out, prev = [], None
    for l in lines:  # drop consecutive duplicates (common in tabbed widgets)
        if l != prev:
            out.append(l)
        prev = l
    return "\n".join(out).strip()


def fetch(url: str) -> dict | None:
    """Return {url, title, fetched_at, text, links, status} or None if blocked."""
    global _last_fetch
    url = normalize_url(url)
    if not allowed(url):
        return None
    wait = CRAWL_DELAY_S - (time.time() - _last_fetch)  # rate limit across all callers
    if wait > 0:
        time.sleep(wait)
    _last_fetch = time.time()
    r = _session.get(url, timeout=30)
    if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
        return {"url": url, "status": r.status_code, "text": "", "links": [], "title": ""}
    final_url = normalize_url(r.url)  # many IRCC links redirect; gold and the index store the final URL
    soup = BeautifulSoup(r.text, "html.parser")
    h1 = soup.find("h1")
    raw_title = soup.title.get_text(" ", strip=True) if soup.title else (h1.get_text(" ", strip=True) if h1 else "")
    # <title> ("Express Entry: Proof of funds") rather than <h1> ("Express Entry"): it names the
    # specific page, which is what a citation needs.
    title = " ".join(raw_title.replace(" - Canada.ca", "").split())
    links = []
    for a in soup.find_all("a", href=True):
        u = normalize_url(urljoin(final_url, a["href"]))
        if urlparse(u).netloc == ALLOWED_HOST and u.endswith(".html"):
            links.append(u)
    main = _clean_main(soup)
    text = _main_text(main)
    return {
        "url": final_url,
        "requested_url": url,
        "status": 200,
        "title": title,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "text": text,
        "links": sorted(set(links)),
    }
