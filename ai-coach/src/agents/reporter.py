"""Weekly report generator."""

# src/agents/reporter.py

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.state.schema import CoachState
from src.memory.long_term import LongTermMemory


REPORT_PROMPT = """You are generating a weekly learning progress report for {name}.

User Profile:
- Target role: {target_role}
- Session count this week: {sessions}
- Topics covered: {topics}
- Current plan week: {current_week} of {total_weeks}
- Skills mastered: {mastered}
- Upcoming: {next_topic}

Write an encouraging, structured weekly report with:
1. 🏆 Wins this week (specific accomplishments)
2. 📈 Progress metrics (% toward goal, velocity)
3. 🎯 Next week's focus (3 specific topics)
4. 📚 Recommended resources (2-3 specific links/titles)
5. 💡 Pro tip for accelerating progress

Keep it under 400 words. Be specific, actionable, and motivating."""


def run_reporter(state: CoachState) -> CoachState:
    """Generate the weekly progress report."""
    llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M", temperature=0.7)
    profile = state["user_profile"]
    plan = state.get("learning_plan")

    prompt = ChatPromptTemplate.from_messages([("human", REPORT_PROMPT)])
    chain = prompt | llm
    result = chain.invoke({
        "name": profile.name,
        "target_role": profile.target_role,
        "sessions": profile.session_count,
        "topics": ", ".join(profile.completed_topics[-5:]) or "Getting started",
        "current_week": plan.current_week if plan else 1,
        "total_weeks": len(plan.weeks) if plan else "TBD",
        "mastered": ", ".join(profile.current_skills[:3]) or "Building foundations",
        "next_topic": plan.weeks[plan.current_week].get("topic", "Next steps") if plan and plan.current_week < len(plan.weeks) else "Advanced topics",
    })

    return {**state, "weekly_report": result.content}