"""
reddit_scraper.py — Scrapes Reddit posts and top comments for Dubai and Bali
travel intelligence using PRAW (Python Reddit API Wrapper).

Targets: r/bali, r/dubai, r/travel, r/solotravel, r/shoestring, r/digitalnomad
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import praw
from praw.models import Submission
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from travel_brain import config

logger = logging.getLogger(__name__)

# ── Reddit Client ─────────────────────────────────────────────────────────────

def _get_reddit_client() -> praw.Reddit:
    if not config.REDDIT_CLIENT_ID or not config.REDDIT_CLIENT_SECRET:
        raise ValueError(
            "Reddit API credentials missing. Set REDDIT_CLIENT_ID and "
            "REDDIT_CLIENT_SECRET in your .env file.\n"
            "Create an app at: https://www.reddit.com/prefs/apps"
        )
    return praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
    )


# ── Extract Post Data ─────────────────────────────────────────────────────────

def extract_post(submission: Submission, location: str, query_used: str = "") -> Optional[dict]:
    """
    Extract text content from a Reddit post + top comments.
    Returns None if the post has no useful text content.
    """
    # Skip removed / deleted posts
    if submission.selftext in ("[removed]", "[deleted]", "") and not submission.url:
        return None

    # Build composite text: title + body + top comments
    body_text = submission.selftext.strip() if submission.selftext else ""

    # Fetch top comments (up to 10, sorted by top)
    try:
        submission.comment_sort = "top"
        submission.comments.replace_more(limit=0)  # Don't deep-load MoreComments
        top_comments = []
        for comment in submission.comments[:10]:
            if (
                hasattr(comment, "body")
                and comment.body not in ("[removed]", "[deleted]")
                and len(comment.body.strip()) > 30
                and comment.score >= 2
            ):
                top_comments.append(comment.body.strip())
    except Exception as e:
        logger.debug(f"Could not load comments for {submission.id}: {e}")
        top_comments = []

    # Combine all text into one rich document
    combined_text = f"TITLE: {submission.title}\n\n"
    if body_text:
        combined_text += f"POST: {body_text}\n\n"
    if top_comments:
        combined_text += "TOP COMMENTS:\n" + "\n---\n".join(top_comments)

    if len(combined_text.strip()) < 100:
        return None  # Too short to be useful

    return {
        "doc_id":        None,
        "source_type":   "reddit",
        "source_url":    f"https://reddit.com{submission.permalink}",
        "source_title":  submission.title,
        "subreddit":     submission.subreddit.display_name,
        "author":        str(submission.author) if submission.author else "[deleted]",
        "score":         submission.score,
        "num_comments":  submission.num_comments,
        "created_utc":   datetime.utcfromtimestamp(submission.created_utc).isoformat() + "Z",
        "location":      location,
        "query_used":    query_used,
        "raw_text":      combined_text,
        "scraped_at":    datetime.utcnow().isoformat() + "Z",
    }


# ── Scrape Subreddit ──────────────────────────────────────────────────────────

def scrape_subreddit(
    reddit: praw.Reddit,
    subreddit_name: str,
    location: str,
    search_terms: list[str],
    limit_per_term: int = 20,
    min_score: int = 5,
) -> list[dict]:
    """Search a subreddit for each keyword and collect matching posts."""
    results = []
    seen_ids = set()

    subreddit = reddit.subreddit(subreddit_name)

    for term in search_terms:
        query = f"{term} {location}"
        logger.info(f"  Searching r/{subreddit_name} for: '{query}'")

        try:
            submissions = subreddit.search(
                query,
                sort="relevance",
                time_filter="year",   # Last 12 months for freshness
                limit=limit_per_term,
            )

            for submission in submissions:
                if submission.id in seen_ids:
                    continue
                if submission.score < min_score:
                    continue
                seen_ids.add(submission.id)

                doc = extract_post(submission, location, query_used=query)
                if doc:
                    results.append(doc)
                time.sleep(0.3)  # Polite rate limiting

        except Exception as e:
            logger.error(f"  Error searching r/{subreddit_name} for '{query}': {e}")

        time.sleep(1)  # Between search terms

    return results


# ── Hot Posts from Location Subreddit ─────────────────────────────────────────

def scrape_hot_posts(reddit: praw.Reddit, subreddit_name: str, location: str, limit: int = 25) -> list[dict]:
    """Grab currently hot posts from a location-specific subreddit (e.g. r/bali)."""
    results = []
    seen_ids = set()

    try:
        subreddit = reddit.subreddit(subreddit_name)
        logger.info(f"  Fetching hot posts from r/{subreddit_name}")

        for submission in subreddit.hot(limit=limit):
            if submission.id in seen_ids:
                continue
            if submission.score < 3:
                continue
            seen_ids.add(submission.id)

            doc = extract_post(submission, location, query_used="hot_feed")
            if doc:
                results.append(doc)
            time.sleep(0.3)

    except Exception as e:
        logger.error(f"  Error fetching hot posts from r/{subreddit_name}: {e}")

    return results


# ── Main Scraper ───────────────────────────────────────────────────────────────

def scrape_reddit(location: str, limit: int = config.SCRAPE_LIMIT) -> list[dict]:
    """
    Full Reddit scrape for a given location.
    Searches general travel subreddits with keywords + pulls hot posts from
    location-specific subreddits (r/bali, r/dubai).
    """
    reddit = _get_reddit_client()
    subreddits   = config.REDDIT_SUBREDDITS.get(location, [])
    search_terms = config.REDDIT_SEARCH_TERMS.get(location, [])

    all_docs = []
    seen_urls = set()
    per_sub   = max(1, limit // len(subreddits)) if subreddits else limit

    # Location-specific subreddit — scrape HOT posts + keyword search
    location_sub = location  # e.g. "bali" or "dubai"
    hot_docs = scrape_hot_posts(reddit, location_sub, location, limit=30)
    for doc in hot_docs:
        if doc["source_url"] not in seen_urls:
            seen_urls.add(doc["source_url"])
            all_docs.append(doc)

    # General travel subreddits — keyword search only
    for sub in subreddits:
        if sub == location_sub:
            continue  # Already scraped above
        docs = scrape_subreddit(
            reddit, sub, location, search_terms,
            limit_per_term=max(3, per_sub // len(search_terms)),
            min_score=5,
        )
        for doc in docs:
            if doc["source_url"] not in seen_urls:
                seen_urls.add(doc["source_url"])
                all_docs.append(doc)

    logger.info(f"Reddit scrape complete for '{location}': {len(all_docs)} unique documents")
    return all_docs


# ── Save Raw Output ────────────────────────────────────────────────────────────

def save_raw(location: str, documents: list[dict]) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = config.RAW_DIR / f"reddit_{location}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(documents)} raw Reddit docs → {out_path}")
    return out_path


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Reddit Travel Scraper")
    parser.add_argument("--location", choices=config.LOCATIONS, default="bali")
    parser.add_argument("--limit", type=int, default=config.SCRAPE_LIMIT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    docs = scrape_reddit(args.location, args.limit)
    if args.dry_run:
        for d in docs[:3]:
            print(f"\n{'='*60}")
            print(f"Subreddit : r/{d['subreddit']}")
            print(f"Title     : {d['source_title']}")
            print(f"Score     : {d['score']}")
            print(f"Text      : {d['raw_text'][:400]}...")
    else:
        save_raw(args.location, docs)
        print(f"\n✅ Saved {len(docs)} Reddit documents for '{args.location}'")
