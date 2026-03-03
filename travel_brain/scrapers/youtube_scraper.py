"""
youtube_scraper.py — Extracts transcripts from YouTube videos relevant to
Dubai and Bali travel using YouTube Data API v3 search + youtube-transcript-api.
"""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from travel_brain import config

logger = logging.getLogger(__name__)


# ── YouTube API Client ────────────────────────────────────────────────────────

def _get_youtube_client():
    if not config.YOUTUBE_API_KEY:
        raise ValueError(
            "YOUTUBE_API_KEY is not set. Add it to your .env file.\n"
            "Get a free key at: https://console.cloud.google.com/"
        )
    return build("youtube", "v3", developerKey=config.YOUTUBE_API_KEY)


# ── Search Videos ─────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_videos(query: str, max_results: int = 10) -> list[dict]:
    """Search YouTube for videos matching a query. Returns list of video metadata."""
    youtube = _get_youtube_client()
    try:
        response = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results,
            videoDuration="medium",        # 4–20 min — ideal for vlogs
            relevanceLanguage="en",
            order="relevance",
            publishedAfter="2023-01-01T00:00:00Z",  # Keep content fresh
        ).execute()

        videos = []
        for item in response.get("items", []):
            snippet = item["snippet"]
            videos.append({
                "video_id":      item["id"]["videoId"],
                "title":         snippet["title"],
                "channel":       snippet["channelTitle"],
                "published_at":  snippet["publishedAt"],
                "description":   snippet["description"][:500],
                "thumbnail":     snippet["thumbnails"]["high"]["url"],
                "query_used":    query,
            })
        logger.info(f"Found {len(videos)} videos for query: '{query}'")
        return videos

    except HttpError as e:
        logger.error(f"YouTube API error: {e}")
        raise


# ── Fetch Transcript ───────────────────────────────────────────────────────────

def fetch_transcript(video_id: str) -> Optional[str]:
    """Fetch and concatenate the transcript for a given YouTube video ID."""
    try:
        # v1.x API: instantiate then call .fetch()
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        # Each snippet has a .text attribute
        full_text = " ".join(
            snippet.text.replace("\n", " ").strip()
            for snippet in transcript
            if snippet.text.strip()
        )
        logger.info(f"Transcript fetched for video {video_id} — {len(full_text)} chars")
        return full_text

    except TranscriptsDisabled:
        logger.warning(f"Transcripts disabled for video {video_id}")
        return None
    except NoTranscriptFound:
        logger.warning(f"No English transcript found for video {video_id}")
        return None
    except Exception as e:
        logger.error(f"Error fetching transcript for {video_id}: {e}")
        return None


# ── Main Scraper ───────────────────────────────────────────────────────────────

def scrape_youtube(location: str, limit: int = config.SCRAPE_LIMIT) -> list[dict]:
    """
    Full YouTube scrape for a given location (e.g. 'bali' or 'dubai').
    Returns a list of structured raw documents ready for the processing pipeline.
    """
    queries = config.YOUTUBE_QUERIES.get(location, [])
    if not queries:
        logger.warning(f"No YouTube queries configured for location: {location}")
        return []

    results = []
    seen_video_ids = set()
    per_query = max(1, limit // len(queries))

    for query in queries:
        try:
            videos = search_videos(query, max_results=per_query)
        except Exception as e:
            logger.error(f"Failed to search YouTube for '{query}': {e}")
            continue

        for video in videos:
            vid_id = video["video_id"]
            if vid_id in seen_video_ids:
                continue
            seen_video_ids.add(vid_id)

            transcript = fetch_transcript(vid_id)
            if not transcript:
                logger.warning(f"Using description as fallback for {vid_id} (Transcript failed/blocked)")
                # Clean up description to be used as fallback text
                transcript = video["description"].replace("\n", " ").strip()
                if len(transcript) < 50:
                    continue  # Too short to be useful

            doc = {
                "doc_id":        None,          # assigned by metadata_builder
                "source_type":   "youtube",
                "source_url":    f"https://www.youtube.com/watch?v={vid_id}",
                "source_title":  video["title"],
                "channel":       video["channel"],
                "published_at":  video["published_at"],
                "location":      location,
                "query_used":    video["query_used"],
                "raw_text":      transcript,
                "scraped_at":    datetime.utcnow().isoformat() + "Z",
            }
            results.append(doc)
            time.sleep(random.uniform(1.0, 2.5))   # Polite rate limiting with jitter to avoid IP bans

        time.sleep(random.uniform(2.0, 4.0))  # Between queries

    logger.info(f"YouTube scrape complete for '{location}': {len(results)} documents")
    return results


# ── Save Raw Output ────────────────────────────────────────────────────────────

def save_raw(location: str, documents: list[dict]) -> Path:
    """Persist raw scraped documents to data/raw/ as JSON."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = config.RAW_DIR / f"youtube_{location}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(documents)} raw YouTube docs → {out_path}")
    return out_path


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="YouTube Travel Scraper")
    parser.add_argument("--location", choices=config.LOCATIONS, default="bali")
    parser.add_argument("--limit", type=int, default=config.SCRAPE_LIMIT)
    parser.add_argument("--dry-run", action="store_true", help="Print results, don't save")
    args = parser.parse_args()

    docs = scrape_youtube(args.location, args.limit)
    if args.dry_run:
        for d in docs[:3]:
            print(f"\n{'='*60}")
            print(f"Title : {d['source_title']}")
            print(f"URL   : {d['source_url']}")
            print(f"Text  : {d['raw_text'][:300]}...")
    else:
        save_raw(args.location, docs)
        print(f"\n✅ Saved {len(docs)} YouTube documents for '{args.location}'")
