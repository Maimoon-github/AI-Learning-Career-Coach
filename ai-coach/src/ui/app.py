"""Chainlit entrypoint."""

# src/ui/app.py

import chainlit as cl
import os
from langchain_core.messages import HumanMessage
from src.graph.builder import build_coach_graph
from src.memory.long_term import LongTermMemory
from src.state.schema import CoachState, UserProfile
from src.voice.stt import WhisperSTT
from src.voice.tts import KokoroTTS
import uuid

# Build graph once at startup
coach_graph = build_coach_graph()
memory_store = LongTermMemory()
stt = WhisperSTT()
tts = KokoroTTS()


@cl.on_chat_start
async def on_chat_start():
    """Initialize session when a new user connects."""
    session_id = str(uuid.uuid4())
    user_id = cl.user_session.get("user_id", session_id)

    # Load or create user profile
    profile = memory_store.load_profile(user_id)
    is_new = profile is None
    if is_new:
        profile = UserProfile(user_id=user_id, name="Learner", target_role="Software Engineer")

    # Initial state
    initial_state: CoachState = {
        "messages": [],
        "user_input": "",
        "voice_mode": False,
        "user_profile": profile,
        "session_id": session_id,
        "is_new_user": is_new,
        "learning_plan": None,
        "current_project": None,
        "evaluation": None,
        "weekly_report": None,
        "next_agent": "",
        "iteration_count": 0,
        "max_iterations": 5,
        "retrieved_docs": [],
        "rag_query": "",
        "hitl_required": False,
        "hitl_prompt": "",
        "human_approval": None,
        "handover_log": [],
        "new_notes": [],
        "finetune_trigger": False,
        "error_count": 0,
        "last_error": None,
    }

    cl.user_session.set("state", initial_state)
    cl.user_session.set("thread_id", session_id)
    cl.user_session.set("user_id", user_id)

    greeting = "👋 Hello! I'm your personalized AI learning coach.\n\nTo get started, tell me:\n1. Your **target role or skill** (e.g., 'ML Engineer', 'learn Rust')\n2. Your GitHub username (optional)\n3. How you prefer to learn (videos / reading / projects)\n\nYou can also say `/voice` to switch to voice mode, or `/report` for your weekly progress summary."
    await cl.Message(content=greeting).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user message."""
    state: CoachState = cl.user_session.get("state")
    thread_id: str = cl.user_session.get("thread_id")

    # ── Command shortcuts ─────────────────────────────────────────
    if message.content.strip().lower() == "/voice":
        state["voice_mode"] = True
        await cl.Message(content="🎙️ Voice mode enabled. Click the microphone to speak.").send()
        return

    if message.content.strip().lower() == "/report":
        state["next_agent"] = "reporter"

    # ── Update state with user input ──────────────────────────────
    state["user_input"] = message.content
    state["messages"] = state["messages"] + [HumanMessage(content=message.content)]
    state["iteration_count"] = 0  # reset per turn

    # ── Stream through the graph ──────────────────────────────────
    config = {"configurable": {"thread_id": thread_id}}
    response_text = ""

    async with cl.Step(name="Thinking...", type="run") as step:
        async for event in coach_graph.astream_events(state, config=config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    response_text += chunk
                    await step.stream_token(chunk)

    # ── Handle HITL interrupt ─────────────────────────────────────
    current_state = coach_graph.get_state(config)
    if current_state.next and "hitl" in current_state.next:
        await _handle_hitl(current_state.values, config)
        return

    # ── Send response ─────────────────────────────────────────────
    if response_text:
        await cl.Message(content=response_text).send()

    # ── TTS if voice mode ─────────────────────────────────────────
    if state.get("voice_mode") and response_text:
        await cl.make_async(tts.speak)(response_text[:500])  # cap TTS length

    # ── Persist updated state ─────────────────────────────────────
    updated_state = coach_graph.get_state(config).values
    cl.user_session.set("state", updated_state)


@cl.action_callback("approve_hitl")
async def approve_hitl(action: cl.Action):
    """User approved the HITL checkpoint."""
    config = {"configurable": {"thread_id": cl.user_session.get("thread_id")}}
    coach_graph.update_state(config, {"human_approval": True, "hitl_required": False})
    await coach_graph.ainvoke(None, config=config)
    await cl.Message(content="✅ Approved! Continuing...").send()


@cl.action_callback("reject_hitl")
async def reject_hitl(action: cl.Action):
    """User rejected — route back to supervisor for retry."""
    config = {"configurable": {"thread_id": cl.user_session.get("thread_id")}}
    coach_graph.update_state(config, {"human_approval": False, "hitl_required": False})
    await coach_graph.ainvoke(None, config=config)
    await cl.Message(content="↩️ Got it! I'll revise the output.").send()


async def _handle_hitl(state, config):
    """Present HITL approval buttons to the user."""
    actions = [
        cl.Action(name="approve_hitl", value="approve", label="✅ Approve"),
        cl.Action(name="reject_hitl",  value="reject",  label="❌ Revise"),
    ]
    await cl.Message(
        content=f"⚠️ **Review Required**\n\n{state.get('hitl_prompt', 'Please review the generated content.')}",
        actions=actions,
    ).send()