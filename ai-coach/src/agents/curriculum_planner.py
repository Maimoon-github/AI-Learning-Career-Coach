"""CrewAI Agent: learning path design."""

# src/agents/curriculum_planner.py

from crewai import Agent, Task, Crew
from src.tools.resource_finder import find_learning_resources
from src.tools.web_search_tool import duckduckgo_search
from src.state.schema import CoachState, LearningPlan

curriculum_planner = Agent(
    role="Adaptive Curriculum Designer & Learning Path Architect",
    goal=(
        "Design a personalized, week-by-week learning roadmap that efficiently bridges "
        "the user's current skill level to their target role. Prioritize hands-on resources, "
        "respect the user's preferred learning style, and adapt the pace to their learning velocity."
    ),
    backstory=(
        "You are a former university professor turned EdTech curriculum designer with deep expertise "
        "in spaced repetition, the Feynman technique, and project-based learning. You've designed "
        "curricula for 50,000+ learners across bootcamps and corporate training programs. "
        "You know that the best path is never a straight line — you sequence topics to build "
        "intuition before formalism, and always anchor abstract concepts in concrete projects."
    ),
    tools=[find_learning_resources, duckduckgo_search],
    llm="ollama/llama3.1:8b-instruct-q4_K_M",
    verbose=True,
    max_iter=4,
    allow_delegation=False,
)


def run_curriculum_planner(state: CoachState) -> CoachState:
    """LangGraph node that invokes the Curriculum Planner crew."""
    profile = state["user_profile"]
    rag_context = "\n".join([d.get("content", "") for d in state["retrieved_docs"][:5]])

    task = Task(
        description=(
            f"Design a {90}-day personalized learning plan for {profile.name}. "
            f"Target role: {profile.target_role}. "
            f"Current skills: {', '.join(profile.current_skills)}. "
            f"Gaps to close: {', '.join(profile.skill_gaps)}. "
            f"Preferred resources: {', '.join(profile.preferred_resources) or 'any'}. "
            f"Learning velocity multiplier: {profile.learning_velocity}. "
            f"Relevant course context from knowledge base:\n{rag_context}\n\n"
            "Output a JSON LearningPlan with: weeks (array of week objects with topic, "
            "resources, project_hint), milestones, estimated_completion_days."
        ),
        expected_output="JSON LearningPlan object",
        agent=curriculum_planner,
    )
    crew = Crew(agents=[curriculum_planner], tasks=[task], verbose=False)
    result = crew.kickoff()

    import json, re
    raw = result.raw
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        plan_data = json.loads(json_match.group())
        plan = LearningPlan(**plan_data)
        return {**state, "learning_plan": plan}
    return state