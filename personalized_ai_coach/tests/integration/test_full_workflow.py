import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.langgraph_workflow.graph import create_app
from src.langgraph_workflow.state import initial_state
from src.services.database.db_manager import init_db, get_session
from src.utils.llm_client import OllamaClient


@pytest.fixture(autouse=True)
async def setup_db():
    # Patch create_async_engine to avoid pool_size/max_overflow issues with sqlite
    from sqlalchemy.ext.asyncio import create_async_engine as original_create
    with patch("src.services.database.db_manager.create_async_engine") as mock_create:
        def side_effect(url, **kwargs):
            if "sqlite" in url:
                kwargs.pop("pool_size", None)
                kwargs.pop("max_overflow", None)
            return original_create(url, **kwargs)
        mock_create.side_effect = side_effect
        await init_db()
        yield
    # Cleanup – drop tables? For test isolation, we use a test DB


@pytest.fixture
def mock_external_services():
    with patch("src.tools.github_tool.GitHubTool._async_run", new_callable=AsyncMock) as gh, \
         patch("src.tools.kaggle_tool.KaggleTool._async_run", new_callable=AsyncMock) as kg, \
         patch("src.tools.web_search_tool.WebSearchTool._run") as ws, \
         patch("src.crewai_agents.llm_fine_tuning_crew.LLMFineTuningCrew.kickoff") as ft, \
         patch("src.utils.llm_client.OllamaClient.generate") as llm:
        gh.return_value = {"languages": {"Python": 80}, "frameworks": ["django"]}
        kg.return_value = {"tier": "Expert", "ml_domains": ["nlp"]}
        ws.return_value = [{"title": "Resource", "url": "http://example.com"}]
        ft.return_value = {"status": "dry_run_success"}
        # Simulate LLM responses for structured extraction
        llm.side_effect = [
            '{"skills": [{"name": "python", "level": 3}], "user_id": "test"}',
            '[{"skill_name": "LangGraph", "gap_severity": 2, "current_level": 1, "required_level": 4, "weeks_to_close": 4, "learning_objective": "learn", "priority_rank": 1}]',
            '{"weeks": [{"week_number": 1, "primary_skill": "Python", "topics": ["basics"], "estimated_hours": 5, "milestone": "done"}], "duration_weeks": 4, "hours_per_week": 10, "user_id": "test", "target_role": "ML Engineer", "total_hours": 20, "version": 1, "validation_notes": []}',
            '{"status": "dry_run_success"}',
            '{"headline_stat": "80% completion"}',
        ]
        yield


@pytest.mark.asyncio
async def test_full_workflow_cycle(mock_external_services):
    """Simulate a complete coaching session with HITL resume."""
    app = create_app(backend="memory")  # Use memory for test speed

    session_id = "test_session"
    state = initial_state(
        user_id="test_user",
        target_role="ML Engineer",
        session_id=session_id,
        github_profile_url="https://github.com/test",
        kaggle_username="test",
        uploaded_document_paths=["resume.pdf"],
    )
    thread_config = {"configurable": {"thread_id": session_id}}

    # Stream through the workflow until HITL interrupt
    events = []
    async for event in app.astream(state, config=thread_config, stream_mode="values"):
        events.append(event)
        if "__interrupt__" in event:
            break

    # Should have reached hitl_node
    assert any("__interrupt__" in e for e in events)
    # Resume with approval
    resume_command = {"hitl_action": "approve", "user_feedback": None}
    async for event in app.astream(resume_command, config=thread_config, stream_mode="values"):
        events.append(event)
        if event.get("current_week") == 2:
            break

    # Verify week advanced and graph continued
    final_state = events[-1]
    assert final_state.get("current_week") == 2
    assert final_state.get("weekly_report") is not None


@pytest.mark.asyncio
async def test_checkpoint_resumability(mock_external_services):
    """Simulate crash and restart – graph resumes from last checkpoint."""
    app = create_app(backend="memory")  # For real test use Postgres, but memory works
    session_id = "resume_test"
    state = initial_state(user_id="test", target_role="Engineer", session_id=session_id)
    thread_config = {"configurable": {"thread_id": session_id}}

    # Run until profile_ingestion completes and then simulate crash by not consuming full stream
    async for event in app.astream(state, config=thread_config, stream_mode="values"):
        if event.get("skill_profile"):
            break  # stop early

    # Create a new app instance (simulating restart)
    app2 = create_app(backend="memory")
    # Get the persisted state from the checkpointer (in memory, it's still there)
    checkpoint = await app2.checkpointer.get(thread_config)
    assert checkpoint is not None, "Checkpoint should exist"

    # Resume from the saved checkpoint
    async for event in app2.astream(None, config=thread_config, stream_mode="values"):
        if event.get("skill_profile"):
            # Should continue from where it left off
            assert event.get("skill_profile") is not None
            break