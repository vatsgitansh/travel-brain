"""
scheduler.py — Automated weekly pipeline re-runs using APScheduler.
Keeps our Travel Brain data fresh without manual intervention.

Run this as a long-lived background process:
  python travel_brain/scheduler.py

Schedule:
  - Bali  Reddit : Every Monday  06:00 UTC
  - Dubai Reddit : Every Monday  06:30 UTC
  - Bali  YouTube: Every Tuesday 06:00 UTC
  - Dubai YouTube: Every Tuesday 06:30 UTC
  - Bali  Blogs  : Every Wednesday 06:00 UTC
  - Dubai Blogs  : Every Wednesday 06:30 UTC
"""

import logging
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))
from travel_brain import config
from travel_brain.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger   = logging.getLogger("scheduler")
console  = Console()
scheduler = BlockingScheduler(timezone="UTC")


def make_job(location: str, source: str):
    """Factory: returns a callable pipeline job for a specific location + source."""
    def job():
        logger.info(f"⏰ Scheduled job starting: {location}/{source}")
        try:
            run_pipeline(
                locations=[location],
                sources=[source],
                limit=config.SCRAPE_LIMIT,
                dry_run=False,                  # Always write in scheduled mode
            )
            logger.info(f"✅ Scheduled job complete: {location}/{source}")
        except Exception as e:
            logger.error(f"❌ Scheduled job failed: {location}/{source} — {e}", exc_info=True)
    job.__name__ = f"pipeline_{location}_{source}"
    return job


def register_jobs():
    """Register all weekly scraping jobs."""
    schedule = [
        # (location, source, day_of_week, hour)
        ("bali",  "reddit",  "mon", 6),
        ("dubai", "reddit",  "mon", 6),
        ("bali",  "youtube", "tue", 6),
        ("dubai", "youtube", "tue", 6),
        ("bali",  "blog",    "wed", 6),
        ("dubai", "blog",    "wed", 6),
    ]

    for location, source, day, hour in schedule:
        minute_offset = 30 if location == "dubai" else 0
        scheduler.add_job(
            make_job(location, source),
            trigger=CronTrigger(day_of_week=day, hour=hour, minute=minute_offset),
            id=f"pipeline_{location}_{source}",
            name=f"Travel Brain — {location.title()} {source.title()}",
            misfire_grace_time=3600,  # Allow up to 1hr late start
            max_instances=1,
        )
        logger.info(
            f"Registered: {location}/{source} "
            f"→ every {day.title()} {hour:02d}:{minute_offset:02d} UTC"
        )


def main():
    console.print("[bold green]🕐 Travel Brain Scheduler Starting...[/]")
    register_jobs()

    console.print(f"\n[bold]Registered {len(scheduler.get_jobs())} scheduled jobs:[/]")
    for job in scheduler.get_jobs():
        console.print(f"  • {job.name}")
        console.print(f"    Next run: {job.next_run_time}")

    console.print("\n[italic]Scheduler running. Press Ctrl+C to stop.[/]")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        console.print("\n[yellow]Scheduler stopped.[/]")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
