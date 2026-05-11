"""Trigger weekly fine-tuning runs."""

# src/finetune/scheduler.py

from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from src.graph.factory import build_graph
from src.state.database import get_session_metadata
from src.state.schema import CoachState
from src.finetune.data_prep import prepare_training_data
from src.finetune.trainer import run_finetune
from src.finetune.export import register_with_ollama


def _weekly_trigger():
    """Run once per week to fine-tune models for all active users."""
    print("⏰ Running weekly fine-tune scheduler...")
    sessions = get_session_metadata(last_days=7)
    users = set(s["user_id"] for s in sessions)

    for user_id in users:
        print(f"  Running fine-tune for {user_id}...")
        data_path = f"./data/training/{user_id}_weekly.jsonl"
        output_dir = f"./data/training/output_weekly_{user_id}"
        timestamp = datetime.utcnow().strftime("%Y%m%d")
        model_name = f"coach-{user_id[:8]}-v{timestamp}"

        n_examples = prepare_training_data(user_id, data_path)
        if n_examples < 10:
            print(f"    Not enough data for {user_id}.")
            continue

        gguf_path = run_finetune(user_id, data_path, output_dir)
        register_with_ollama(gguf_path, model_name)
        print(f"    ✓ Fine-tuned and registered: {model_name}")


def start_scheduler():
    """Start the background scheduler for weekly fine-tuning."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _weekly_trigger,
        "cron",
        day_of_week="mon",  # run every Monday
        hour=3,            # at 3 AM
        minute=0,
        timezone="UTC",
    )
    scheduler.start()
    print("⏰ Weekly fine-tune scheduler started (every Monday 3 AM UTC).")

