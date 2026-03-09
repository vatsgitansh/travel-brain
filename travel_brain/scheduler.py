"""
scheduler.py — Automated biweekly pipeline for Travel Brain.

Runs every 2 weeks (1st and 15th of each month at 03:00 UTC).
Scrapes all sources for all locations and upserts new/updated content
into the ChromaDB vector database to keep the AI's knowledge fresh.

Modes:
  1. Embedded (default):  imported by app.py and runs as a background thread
  2. Standalone:          `python travel_brain/scheduler.py`  (for local dev/cron)
"""

import logging
import sys
import threading
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from travel_brain import config
from travel_brain.pipeline import run_pipeline

logger  = logging.getLogger("scheduler")
console = Console()

# ── Job Definitions ───────────────────────────────────────────────────────────

# Biweekly schedule: runs on the 1st and 15th of each month at 03:00 UTC.
# Sources are staggered by 15 minutes to avoid overloading APIs simultaneously.
BIWEEKLY_JOBS = [
    # (location, source,     hour, minute)
    ("bali",  "reddit",      3,    0),
    ("dubai", "reddit",      3,   15),
    ("bali",  "youtube",     3,   30),
    ("dubai", "youtube",     3,   45),
    ("bali",  "blog",        4,    0),
    ("dubai", "blog",        4,   15),
]


def make_job(location: str, source: str):
    """Factory: returns a callable pipeline job for a specific location + source."""
    def _job():
        logger.info(f"⏰ Biweekly pipeline starting: {location}/{source}")
        try:
            stats = run_pipeline(
                locations=[location],
                sources=[source],
                limit=config.SCRAPE_LIMIT,
                dry_run=False,
            )
            upserted = stats.get("upserted", 0) if isinstance(stats, dict) else "?"
            logger.info(f"✅ Pipeline done: {location}/{source} — {upserted} chunks upserted")
        except Exception as e:
            logger.error(
                f"❌ Pipeline failed: {location}/{source} — {e}",
                exc_info=True,
            )
    _job.__name__ = f"pipeline_{location}_{source}"
    return _job


def _build_scheduler(blocking: bool = False):
    """Create and register all biweekly jobs. Returns the configured scheduler."""
    sched = BlockingScheduler(timezone="UTC") if blocking else BackgroundScheduler(timezone="UTC")

    for location, source, hour, minute in BIWEEKLY_JOBS:
        sched.add_job(
            make_job(location, source),
            # day='1,15' → 1st and 15th of each month
            trigger=CronTrigger(day="1,15", hour=hour, minute=minute),
            id=f"biweekly_{location}_{source}",
            name=f"Travel Brain — {location.title()} {source.title()} (biweekly)",
            misfire_grace_time=7200,   # Allow up to 2 hrs late if server was down
            max_instances=1,
            replace_existing=True,
        )

    return sched


# ── Embedded Mode (called from app.py lifespan) ───────────────────────────────

_scheduler_instance = None
_scheduler_lock = threading.Lock()


def start_background_scheduler() -> None:
    """
    Start the biweekly scheduler as a background thread.
    Safe to call multiple times — only starts once.
    Called from FastAPI's lifespan handler.
    """
    global _scheduler_instance
    with _scheduler_lock:
        if _scheduler_instance is not None:
            return
        sched = _build_scheduler(blocking=False)
        sched.start()
        _scheduler_instance = sched

        # Print next run times
        table = Table(title="📅 Biweekly Scraping Schedule (UTC)", show_lines=True)
        table.add_column("Job", style="cyan")
        table.add_column("Next Run", style="green")
        for job in sched.get_jobs():
            table.add_row(job.name, str(job.next_run_time))
        console.print(table)
        logger.info(f"Biweekly scheduler started with {len(sched.get_jobs())} jobs.")


def stop_background_scheduler() -> None:
    """Gracefully shut down the scheduler (called on app shutdown)."""
    global _scheduler_instance
    with _scheduler_lock:
        if _scheduler_instance and _scheduler_instance.running:
            _scheduler_instance.shutdown(wait=False)
            _scheduler_instance = None
            logger.info("Biweekly scheduler stopped.")


# ── Standalone Mode ───────────────────────────────────────────────────────────

def main():
    """Run the scheduler as a standalone blocking process (for local dev/cron)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    console.print("[bold green]🕐 Travel Brain Biweekly Scheduler Starting...[/]")
    sched = _build_scheduler(blocking=True)

    table = Table(title="📅 Registered Jobs", show_lines=True)
    table.add_column("Job")
    table.add_column("Next Run")
    for job in sched.get_jobs():
        table.add_row(job.name, str(job.next_run_time))
    console.print(table)
    console.print("\n[italic]Scheduler running. Press Ctrl+C to stop.[/]")

    try:
        sched.start()
    except KeyboardInterrupt:
        console.print("\n[yellow]Scheduler stopped by user.[/]")
        sched.shutdown()


if __name__ == "__main__":
    main()
