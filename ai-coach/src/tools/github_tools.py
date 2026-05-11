"""GitHub API tools."""

# src/tools/github_tools.py

from crewai_tools import tool
from github import Github
import os


@tool("analyze_github_profile")
def analyze_github_profile(username: str) -> dict:
    """
    Analyze a GitHub profile to extract programming languages, 
    repo complexity, commit frequency, and project diversity.
    Returns a structured dict of technical signals.
    """
    g = Github(os.environ.get("GITHUB_TOKEN"))
    user = g.get_user(username)
    repos = list(user.get_repos(type="owner", sort="updated"))[:20]

    languages = {}
    for repo in repos:
        if repo.language:
            languages[repo.language] = languages.get(repo.language, 0) + 1

    return {
        "username": username,
        "public_repos": user.public_repos,
        "top_languages": sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5],
        "followers": user.followers,
        "recent_repos": [
            {
                "name": r.name,
                "language": r.language,
                "stars": r.stargazers_count,
                "description": r.description or "",
            }
            for r in repos[:10]
        ],
    }