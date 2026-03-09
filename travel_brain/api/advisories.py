from fastapi import APIRouter
import xml.etree.ElementTree as ET
import urllib.request
from datetime import datetime
import threading
import logging

router = APIRouter(prefix="/advisories", tags=["Advisories"])
logger = logging.getLogger(__name__)

# Map our internal locations to US State Dept Country Names
LOCATION_MAP = {
    "bali": "Indonesia",
    "dubai": "United Arab Emirates",
}

# Cache to avoid hitting the RSS feed on every request
_cache = {
    "data": {},
    "last_fetched": None
}
_cache_lock = threading.Lock()

def fetch_advisories():
    """Fetch live travel advisories from US State Dept RSS feed."""
    url = "https://travel.state.gov/_res/rss/TAsTWs.xml"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            
        root = ET.fromstring(xml_data)
        advisories = {}
        
        # Parse the RSS Feed
        for item in root.findall('./channel/item'):
            title_el = item.find('title')
            desc_el = item.find('description')
            link_el = item.find('link')
            
            title = str(title_el.text) if title_el is not None and title_el.text else ""
            desc = str(desc_el.text) if desc_el is not None and desc_el.text else ""
            link = str(link_el.text) if link_el is not None and link_el.text else ""
            
            # Title format: "United Arab Emirates - Level 2: Exercise Increased Caution"
            if " - Level " in title:
                country, rest = title.split(" - Level ", 1)
                level_str = rest.split(":")[0]  # "1", "2", "3", "4"
                
                try:
                    level = int(level_str)
                except ValueError:
                    level = 0
                
                # Extract clean text from HTML description cdats
                import re
                clean_desc = re.sub('<[^<]+>', '', desc).replace('&nbsp;', ' ').strip()
                
                advisories[country.strip()] = {
                    "level": level,
                    "title": rest.strip(),
                    "description": clean_desc,
                    "link": link
                }
                
        return advisories
    except Exception as e:
        logger.error(f"Failed to fetch travel advisories: {e}")
        return {}

def get_live_advisories() -> dict:
    """Thread-safe getter with 1-hour cache."""
    with _cache_lock:
        now = datetime.now()
        last_fetched = _cache.get("last_fetched")
        # Refresh if cache is empty or older than 1 hour
        should_fetch = False
        if not _cache["data"] or last_fetched is None:
            should_fetch = True
        elif isinstance(last_fetched, datetime) and (now - last_fetched).total_seconds() > 3600:
            should_fetch = True
            
        if should_fetch:
            logger.info("Fetching fresh travel advisories from State Dept...")
            new_data = fetch_advisories()
            if new_data:
                _cache["data"] = new_data
                _cache["last_fetched"] = now
                
        # To avoid type errors, ensure we return a dictionary
        data = _cache.get("data")
        return data if isinstance(data, dict) else {}

@router.get("/")
async def get_all_advisories():
    """Get live travel advisory status for our tracked destinations from the US State Dept."""
    live_data = get_live_advisories()

    results = {}
    for loc_key, country_name in LOCATION_MAP.items():
        if country_name in live_data:
            results[loc_key] = live_data[country_name]
        else:
            # Advisory data unavailable — show a neutral placeholder, NOT a fake alert
            results[loc_key] = {
                "level": None,
                "title": "Advisory data unavailable",
                "description": f"Could not retrieve advisory data for {country_name}. Please check travel.state.gov directly.",
                "link": "https://travel.state.gov/"
            }

    return results
