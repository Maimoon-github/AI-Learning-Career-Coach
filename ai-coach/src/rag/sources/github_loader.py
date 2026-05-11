"""GitHub source loader."""

# src/rag/sources/github_loader.py

from typing import List, Dict, Any


class GithubLoader:
    """Load GitHub READMEs for learning resources."""
    
    def load(self, repo_urls: List[str]) -> List[Dict[str, Any]]:
        """Load READMEs from GitHub repositories."""
        documents = []
        for url in repo_urls:
            # In production, use GitHub API to fetch README
            # Example: https://api.github.com/repos/{owner}/{repo}/readme
            try:
                documents.append({
                    "text": f"README content from {url} would be fetched here.",
                    "source": url
                })
            except Exception as e:
                print(f"Error loading GitHub repo {url}: {e}")
        return documents
