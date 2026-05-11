"""LangGraph supervisor node."""

# src/agents/supervisor.py

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.state.schema import CoachState

SUPERVISOR_SYSTEM = """You are the orchestration supervisor for an AI learning coach system.
Given the conversation history and the current user profile, decide which specialist agent 
should handle the next action. 

Available agents:
- profile_analyst   : Analyze GitHub/Kaggle/uploaded data; extract skills
- curriculum_planner: Design or update a personalized learning path
- project_builder   : Generate a hands-on practice project
- evaluator         : Assess the quality of a plan or project
- reporter          : Generate the weekly progress report
- responder         : Directly respond to the user (no specialist needed)

Respond with ONLY the agent name. No explanation."""

def supervisor_node(state: CoachState) -> CoachState:
    """Routes to the appropriate specialist based on current state."""
    llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_SYSTEM),
        ("human", "User said: {user_input}\n\nCurrent phase: {phase}\n\nRoute to:"),
    ])
    chain = prompt | llm
    result = chain.invoke({
        "user_input": state["user_input"],
        "phase": _detect_phase(state),
    })
    next_agent = result.content.strip().lower()

    # Cycle guard
    if state["iteration_count"] >= state["max_iterations"]:
        next_agent = "responder"

    return {
        **state,
        "next_agent": next_agent,
        "iteration_count": state["iteration_count"] + 1,
    }


def _detect_phase(state: CoachState) -> str:
    """Heuristic: determine what phase the user is in."""
    if state["is_new_user"] or not state["user_profile"].current_skills:
        return "onboarding"
    if state["learning_plan"] is None:
        return "planning"
    if state["user_profile"].session_count % 7 == 0:
        return "weekly_report"
    return "active_learning"