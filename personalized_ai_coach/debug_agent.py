from crewai import Agent
from src.tools.github_tool import GitHubTool
import pydantic
try:
    tool = GitHubTool()
    agent = Agent(role="r", goal="g", backstory="b", tools=[tool])
    print("Agent with tool ok")
except pydantic.ValidationError as e:
    print(f"Agent validation failed: {e}")
except Exception as e:
    print(f"Other error: {e}")
