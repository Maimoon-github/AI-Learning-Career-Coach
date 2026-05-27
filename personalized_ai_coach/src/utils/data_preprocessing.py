import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pypdf
from markdown_it import MarkdownIt


def clean_github_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize signals from GitHub API response."""
    cleaned = {
        "languages": {},
        "frameworks": [],
        "contribution_streak_days": 0,
        "project_complexity_score": 0.0,
        "key_projects": [],
        "collaboration_signals": {},
    }
    # Simplified extraction – actual implementation would parse specific fields
    if "languages" in raw_data:
        cleaned["languages"] = {lang: float(percent) for lang, percent in raw_data["languages"].items()}
    if "frameworks" in raw_data:
        cleaned["frameworks"] = [f.lower() for f in raw_data["frameworks"]]
    return cleaned


def clean_kaggle_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise Kaggle profile data."""
    cleaned = {
        "tier": raw_data.get("tier", "Novice"),
        "medals": raw_data.get("medals", {}),
        "ml_domains": [d.lower() for d in raw_data.get("ml_domains", [])],
        "notebook_quality_score": float(raw_data.get("notebook_quality_score", 0.0)),
        "active_last_year": raw_data.get("active_last_year", False),
        "strongest_domain": raw_data.get("strongest_domain", ""),
    }
    return cleaned


def extract_text_from_pdf(path: Path) -> str:
    """Extract plain text from a PDF file."""
    text = []
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            if page_text := page.extract_text():
                text.append(page_text)
    return "\n".join(text)


def extract_text_from_markdown(path: Path) -> str:
    """Extract plain text from a Markdown file (strip formatting)."""
    md = MarkdownIt()
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    html = md.render(content)
    # Very basic HTML tag stripping – in production use a proper parser
    text = re.sub(r"<[^>]+>", "", html)
    return text


def extract_text_from_plain(path: Path) -> str:
    """Read plain text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_document_text(path: Path) -> str:
    """Route to the correct extractor based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    elif suffix in (".md", ".markdown"):
        return extract_text_from_markdown(path)
    elif suffix == ".txt":
        return extract_text_from_plain(path)
    else:
        raise ValueError(f"Unsupported document type: {suffix}")


def deduplicate_session_notes(notes: List[str], threshold: float = 0.85) -> List[str]:
    """Remove near-duplicate notes using simple Jaccard similarity."""
    def jaccard(a: str, b: str) -> float:
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a and not set_b:
            return 1.0
        return len(set_a & set_b) / len(set_a | set_b)

    unique = []
    for note in notes:
        if not unique:
            unique.append(note)
            continue
        if all(jaccard(note, existing) < threshold for existing in unique):
            unique.append(note)
    return unique


def hash_document_content(content: str) -> str:
    """Generate SHA-256 hash of document content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()