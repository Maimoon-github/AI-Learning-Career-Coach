"""LLM-as-Judge evaluator node."""

# src/agents/evaluator.py

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from src.state.schema import CoachState, EvaluationResult

EVAL_PROMPT = """You are a strict quality evaluator for an AI tutoring system.

Evaluate the following output on a scale of 0.0–1.0:
- 0.0–0.4: Poor — missing key elements, inaccurate, or not personalized
- 0.4–0.7: Acceptable — correct but generic or missing nuance
- 0.7–0.9: Good — accurate, personalized, actionable
- 0.9–1.0: Excellent — exceeds expectations

Output JSON: {{"score": float, "passed": bool, "feedback": str, "retry_recommended": bool, "escalate_to_human": bool}}

Content to evaluate:
{content}

Evaluation criteria:
{criteria}"""


def evaluator_node(state: CoachState) -> CoachState:
    """LLM-as-Judge node. Evaluates the most recent agent output."""
    llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M", temperature=0)

    # Determine what to evaluate
    if state.get("learning_plan"):
        content = state["learning_plan"].model_dump_json(indent=2)
        criteria = "Is the plan personalized? Is the pacing realistic? Are resources concrete?"
    elif state.get("current_project"):
        content = state["current_project"].model_dump_json(indent=2)
        criteria = "Is the scope achievable in 4-12 hours? Does it match the current topic?"
    else:
        return {**state, "evaluation": EvaluationResult(
            score=1.0, passed=True, feedback="Nothing to evaluate.", retry_recommended=False
        )}

    prompt = ChatPromptTemplate.from_messages([
        ("human", EVAL_PROMPT)
    ])
    chain = prompt | llm
    result = chain.invoke({"content": content, "criteria": criteria})

    import json, re
    raw = result.content
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if json_match:
        eval_data = json.loads(json_match.group())
        evaluation = EvaluationResult(**eval_data)
    else:
        evaluation = EvaluationResult(
            score=0.5, passed=True, feedback="Could not parse evaluation.", retry_recommended=False
        )

    return {
        **state,
        "evaluation": evaluation,
        "hitl_required": evaluation.escalate_to_human,
        "hitl_prompt": f"Evaluator flagged output for review: {evaluation.feedback}" if evaluation.escalate_to_human else "",
    }