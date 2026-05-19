#!/usr/bin/env python3
"""
Personalized AI Learning & Career Coach
Entry point: initializes LangGraph workflow and optional voice interface.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import click
import structlog
import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

load_dotenv()

from src.langgraph_workflow.graph import create_app
from src.langgraph_workflow.state import initial_state

console = Console()


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer() if os.getenv("APP_ENV") == "development"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), os.getenv("LOG_LEVEL", "INFO"))
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


log = structlog.get_logger(__name__)


async def run_coaching_session(
    user_id: str,
    target_role: str,
    github_url: str | None = None,
    kaggle_username: str | None = None,
    document_paths: list[str] | None = None,
    enable_voice: bool = False,
) -> dict[str, Any]:
    """
    Execute a full coaching session via the LangGraph workflow.
    Handles HITL interrupts by prompting the user interactively via CLI or voice.
    """
    settings = yaml.safe_load(open("config/system_settings.yaml"))
    backend = settings["workflow"]["checkpoint_backend"]
    app = create_app(backend=backend)

    session_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": session_id}}
    state = initial_state(
        user_id=user_id,
        target_role=target_role,
        session_id=session_id,
        github_profile_url=github_url,
        kaggle_username=kaggle_username,
        uploaded_document_paths=document_paths or [],
    )

    console.print(Panel(
        f"[bold green]Starting coaching session[/bold green]\n"
        f"User: [cyan]{user_id}[/cyan] | Target Role: [yellow]{target_role}[/yellow]\n"
        f"Session: [dim]{session_id}[/dim]",
        title="AI Learning Coach",
    ))

    voice_stt = voice_tts = None
    if enable_voice:
        from src.services.voice_interface.voice_services import STTService, TTSService
        voice_stt = STTService()
        voice_tts = TTSService()

    final_state: dict[str, Any] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:

        task_id = progress.add_task("Initializing workflow...", total=None)

        # Stream workflow events
        async for event in app.astream(state, config=thread_config, stream_mode="values"):
            # Detect which node just completed
            if "__interrupt__" in event:
                # HITL interrupt: present to user and collect response
                progress.update(task_id, description="[yellow]Awaiting your review...")
                interrupt_data = event["__interrupt__"][0].value
                user_response = await _handle_hitl_interrupt(
                    interrupt_data, voice_stt, voice_tts
                )
                # Resume workflow with user's decision
                async for resumed_event in app.astream(
                    {"hitl_action": user_response["action"], "user_feedback": user_response.get("feedback")},
                    config=thread_config,
                    stream_mode="values",
                ):
                    final_state = resumed_event
            else:
                final_state = event
                _display_node_progress(event, progress, task_id)

    console.print("\n[bold green]✓ Session complete[/bold green]")
    return final_state


async def _handle_hitl_interrupt(
    data: dict[str, Any],
    stt=None,
    tts=None,
) -> dict[str, Any]:
    """Present HITL data to user and collect approve/revise/end response."""
    report = data.get("weekly_report", {})
    week = data.get("current_week", 1)

    console.print(Panel(
        f"[bold]Week {week} Review[/bold]\n\n"
        + _format_report_summary(report),
        title="[yellow]Your Review Required[/yellow]",
        border_style="yellow",
    ))

    if tts and report:
        summary = f"Week {week} complete. {report.get('coach_note', 'Keep up the great work.')}"
        await tts.synthesize(summary)

    while True:
        console.print("\n[cyan]What would you like to do?[/cyan]")
        console.print("  [green][A]pprove[/green] — Continue to next week")
        console.print("  [yellow][R]evise[/yellow] — Adjust the learning path")
        console.print("  [red][E]nd[/red] — End this session")

        if stt:
            console.print("[dim]Speak your choice...[/dim]")
            # Voice input would be captured here
            choice = input("Choice (A/R/E): ").strip().upper()
        else:
            choice = input("Choice (A/R/E): ").strip().upper()

        if choice in ("A", "APPROVE"):
            return {"action": "approve"}
        elif choice in ("R", "REVISE"):
            feedback = input("What should be changed? ").strip()
            return {"action": "revise", "feedback": feedback or None}
        elif choice in ("E", "END"):
            return {"action": "end"}
        else:
            console.print("[red]Invalid choice. Enter A, R, or E.[/red]")


def _format_report_summary(report: dict) -> str:
    if not report:
        return "No report data available."
    lines = []
    if hs := report.get("headline_stat"):
        lines.append(f"🏆 {hs}")
    for win in report.get("wins", []):
        lines.append(f"✅ {win}")
    if note := report.get("coach_note"):
        lines.append(f"\n💬 {note}")
    return "\n".join(lines) if lines else str(report)


def _display_node_progress(event: dict, progress, task_id) -> None:
    """Map workflow state changes to human-readable progress messages."""
    if event.get("skill_profile") and not event.get("skill_gaps"):
        progress.update(task_id, description="[green]✓ Profile analyzed — Assessing skill gaps...")
    elif event.get("skill_gaps") and not event.get("learning_path"):
        progress.update(task_id, description=f"[green]✓ {len(event['skill_gaps'])} gaps found — Building learning path...")
    elif event.get("learning_path") and not event.get("practice_projects"):
        progress.update(task_id, description="[green]✓ Learning path generated — Creating projects...")
    elif event.get("practice_projects"):
        progress.update(task_id, description=f"[green]✓ {len(event['practice_projects'])} projects created — Generating report...")
    elif event.get("weekly_report"):
        progress.update(task_id, description="[green]✓ Weekly report ready")


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Personalized AI Learning & Career Coach CLI."""
    _configure_logging()


@cli.command()
@click.option("--user-id", required=True, help="Unique user identifier")
@click.option("--target-role", required=True, help="Target job role (e.g., 'ML Engineer')")
@click.option("--github", default=None, help="GitHub profile URL")
@click.option("--kaggle", default=None, help="Kaggle username")
@click.option("--docs", multiple=True, help="Paths to resume/notes documents")
@click.option("--voice", is_flag=True, default=False, help="Enable voice interface")
def start(user_id: str, target_role: str, github: str | None, kaggle: str | None, docs: tuple, voice: bool):
    """Start a new coaching session."""
    asyncio.run(
        run_coaching_session(
            user_id=user_id,
            target_role=target_role,
            github_url=github,
            kaggle_username=kaggle,
            document_paths=list(docs),
            enable_voice=voice,
        )
    )


@cli.command()
def health():
    """Check system health (Ollama, database)."""
    async def _check():
        import httpx
        results = {}
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{base_url}/api/tags")
                results["ollama"] = "✓ online" if r.status_code == 200 else f"✗ HTTP {r.status_code}"
        except Exception as e:
            results["ollama"] = f"✗ {e}"

        from src.services.database.db_manager import health_check
        results["database"] = "✓ online" if await health_check() else "✗ offline"

        for service, status in results.items():
            color = "green" if "✓" in status else "red"
            console.print(f"[{color}]{service}: {status}[/{color}]")

    asyncio.run(_check())


if __name__ == "__main__":
    cli()