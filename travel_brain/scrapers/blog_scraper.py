"""
blog_scraper.py — Scrapes niche travel blog content for Dubai and Bali.
Uses requests + BeautifulSoup4 for static pages, with robots.txt compliance,
user-agent rotation, and rate limiting built in.
"""

import hashlib
import json
import logging
import random
import time
import urllib.robotparser
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from travel_brain import config

logger = logging.getLogger(__name__)

# ── User Agent Pool (rotation to avoid basic blocking) ────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

REQUEST_TIMEOUT = 15  # seconds
MIN_TEXT_LENGTH = 300  # characters — skip pages with very little text


# ── Session Factory ───────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
    })
    return session


# ── Robots.txt Compliance ─────────────────────────────────────────────────────

_robots_cache: dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}

def _is_allowed(url: str) -> bool:
    """Check robots.txt before fetching a URL."""
    parsed = urlparse(url)
    base   = f"{parsed.scheme}://{parsed.netloc}"
    if base not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        try:
            rp.set_url(f"{base}/robots.txt")
            rp.read()
            _robots_cache[base] = rp
        except Exception:
            _robots_cache[base] = None  # If robots.txt unreadable, allow

    rp = _robots_cache.get(base)
    if rp is None:
        return True
    return rp.can_fetch("*", url)


# ── Page Fetcher ──────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def fetch_page(url: str, session: Optional[requests.Session] = None) -> Optional[BeautifulSoup]:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    if not _is_allowed(url):
        logger.info(f"  Blocked by robots.txt: {url}")
        return None

    s = session or _make_session()
    try:
        response = s.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        if "text/html" not in response.headers.get("Content-Type", ""):
            return None
        return BeautifulSoup(response.text, "lxml")
    except requests.HTTPError as e:
        logger.warning(f"  HTTP {e.response.status_code} for {url}")
        if e.response.status_code in (403, 404, 410):
            return None   # Don't retry
        raise
    except requests.RequestException as e:
        logger.warning(f"  Request failed for {url}: {e}")
        raise


# ── Article Text Extractor ────────────────────────────────────────────────────

# Tags whose content we should strip (ads, nav, footer, etc.)
_NOISE_TAGS = [
    "nav", "footer", "header", "aside", "script", "style",
    "form", "iframe", "noscript", "advertisement", "cookie",
]

def extract_article_text(soup: BeautifulSoup, url: str) -> Optional[str]:
    """
    Extract the main article text from a BeautifulSoup page.
    Removes navigation, ads, scripts and focuses on <article>, <main>, or <p> tags.
    """
    # Remove noise elements
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()
    for tag in soup.find_all(class_=lambda c: c and any(
        noise in c.lower() for noise in ["nav", "footer", "sidebar", "ad", "cookie", "popup", "newsletter"]
    )):
        tag.decompose()

    # Find the best content container
    article = (
        soup.find("article") or
        soup.find("main") or
        soup.find(class_=lambda c: c and any(
            x in c.lower() for x in ["post-content", "entry-content", "article-body", "blog-content"]
        ))
    )

    if article:
        paragraphs = article.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    text = "\n\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs if len(p.get_text().strip()) > 40)

    if len(text) < MIN_TEXT_LENGTH:
        return None

    return text


def extract_title(soup: BeautifulSoup) -> str:
    """Extract article title from the page."""
    for selector in [
        soup.find("h1"),
        soup.find("meta", {"property": "og:title"}),
        soup.find("title"),
    ]:
        if selector:
            return (
                selector.get("content", "") or selector.get_text()
            ).strip()
    return "Unknown Title"


def extract_published_date(soup: BeautifulSoup) -> Optional[str]:
    """Try to find the article publication date from common meta patterns."""
    for meta in soup.find_all("meta"):
        prop = meta.get("property", "") or meta.get("name", "")
        if prop in ("article:published_time", "datePublished", "pubdate"):
            return meta.get("content", "")
    time_tag = soup.find("time")
    if time_tag:
        return time_tag.get("datetime", time_tag.get_text(strip=True))
    return None


# ── Internal Link Discovery ───────────────────────────────────────────────────

def discover_article_links(soup: BeautifulSoup, base_url: str, location: str) -> list[str]:
    """
    Find links on the page that likely lead to travel articles about our location.
    Filters by URL path and anchor text keywords.
    """
    location_keywords = [location, "bali" if location == "bali" else "dubai", "travel", "guide", "tip"]
    links = set()
    parsed_base = urlparse(base_url)

    for a_tag in soup.find_all("a", href=True):
        href  = a_tag["href"].strip()
        text  = a_tag.get_text(strip=True).lower()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue

        full_url = urljoin(base_url, href)
        parsed   = urlparse(full_url)

        # Stay on the same domain
        if parsed.netloc != parsed_base.netloc:
            continue

        # Must contain a location keyword in URL or anchor text
        url_lower = full_url.lower()
        if any(kw in url_lower or kw in text for kw in location_keywords):
            links.add(full_url)

    link_list = list(links)
    return link_list[:50]  # Cap discovered links per page


# ── Single Blog Scrape ────────────────────────────────────────────────────────

def scrape_single_url(url: str, location: str, session: requests.Session) -> Optional[dict]:
    """Scrape one blog URL and return a structured document."""
    try:
        soup = fetch_page(url, session)
        if not soup:
            return None
    except Exception as e:
        logger.warning(f"  Skipping {url} due to repeated failures: {e}")
        return None

    text = extract_article_text(soup, url)
    if not text:
        logger.debug(f"  No useful text at: {url}")
        return None

    title       = extract_title(soup)
    pub_date    = extract_published_date(soup)

    return {
        "doc_id":       None,
        "source_type":  "blog",
        "source_url":   url,
        "source_title": title,
        "published_at": pub_date,
        "location":     location,
        "raw_text":     text,
        "text_hash":    hashlib.sha256(text.encode()).hexdigest(),
        "scraped_at":   datetime.utcnow().isoformat() + "Z",
    }


# ── Main Blog Scraper ─────────────────────────────────────────────────────────

def scrape_blogs(location: str, limit: int = config.SCRAPE_LIMIT) -> list[dict]:
    """
    Crawl seed blog URLs for a given location, then discover and scrape
    linked articles within the same domain.
    Returns structured document list.
    """
    seeds = config.BLOG_SEEDS.get(location, [])
    if not seeds:
        logger.warning(f"No blog seeds configured for: {location}")
        return []

    session   = _make_session()
    results   = []
    seen_urls = set()
    seen_hashes: set[str] = set()

    # Queue: (url, depth) — depth 0 = seed, depth 1 = discovered article
    queue: list[tuple[str, int]] = [(url, 0) for url in seeds]

    while queue and len(results) < limit:
        url, depth = queue.pop(0)

        if url in seen_urls:
            continue
        seen_urls.add(url)

        logger.info(f"  Scraping (depth={depth}): {url}")

        doc = scrape_single_url(url, location, session)

        if doc:
            if doc["text_hash"] not in seen_hashes:
                seen_hashes.add(doc["text_hash"])
                results.append(doc)
                logger.info(f"    ✓ Collected: '{doc['source_title']}' ({len(doc['raw_text'])} chars)")
            else:
                logger.debug(f"    ⚠ Duplicate content at: {url}")

            # Discover more articles from this page (depth 0 and 1)
            if depth < 2:
                try:
                    soup = fetch_page(url, session)
                    if soup:
                        new_links = discover_article_links(soup, url, location)
                        for link in new_links:
                            if link not in seen_urls:
                                queue.append((link, depth + 1))
                except Exception as e:
                    logger.debug(f"    ⚠ Could not fetch {url} for links: {e}")

        time.sleep(random.uniform(1.5, 3.0))  # Polite, human-like delay

    logger.info(f"Blog scrape complete for '{location}': {len(results)} documents")
    return results


# ── Save Raw Output ────────────────────────────────────────────────────────────

def save_raw(location: str, documents: list[dict]) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path  = config.RAW_DIR / f"blog_{location}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(documents)} raw blog docs → {out_path}")
    return out_path


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Niche Blog Travel Scraper")
    parser.add_argument("--location", choices=config.LOCATIONS, default="bali")
    parser.add_argument("--limit", type=int, default=config.SCRAPE_LIMIT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    docs = scrape_blogs(args.location, args.limit)
    if args.dry_run:
        for d in docs[:3]:
            print(f"\n{'='*60}")
            print(f"Blog  : {d['source_url']}")
            print(f"Title : {d['source_title']}")
            print(f"Text  : {d['raw_text'][:400]}...")
    else:
        save_raw(args.location, docs)
        print(f"\n✅ Saved {len(docs)} blog documents for '{args.location}'")
