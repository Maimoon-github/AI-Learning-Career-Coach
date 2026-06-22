import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pypdf
import structlog
from markdown_it import MarkdownIt

log = structlog.get_logger(__name__)

class DataPreprocessor:
    """
    Consolidated utility for cleaning and normalizing data for LLM consumption.
    Features:
    - GitHub/Kaggle signal normalization.
    - Robust text extraction with error isolation.
    - Token budget enforcement (simple heuristic).
    - Near-duplicate detection.
    """

    def __init__(self, token_budget: int = 4000):
        self.token_budget = token_budget
        self._md = MarkdownIt()

    def clean_github_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize signals from GitHub API response."""
        try:
            cleaned = {
                "languages": {k: float(v) for k, v in raw_data.get("languages", {}).items()},
                "frameworks": sorted(list(set(f.lower() for f in raw_data.get("frameworks", [])))),
                "contribution_streak_days": int(raw_data.get("contribution_streak_days", 0)),
                "project_complexity_score": float(raw_data.get("project_complexity_score", 0.0)),
                "key_projects": raw_data.get("key_projects", [])[:5],  # Limit to top 5
                "collaboration_signals": raw_data.get("collaboration_signals", {}),
            }
            return cleaned
        except (TypeError, ValueError) as e:
            log.error("github_data_cleaning_failed", error=str(e))
            return {"error": "Failed to clean GitHub data"}

    def clean_kaggle_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Kaggle profile data."""
        try:
            cleaned = {
                "tier": raw_data.get("tier", "Novice"),
                "medals": raw_data.get("medals", {}),
                "ml_domains": sorted(list(set(d.lower() for d in raw_data.get("ml_domains", [])))),
                "notebook_quality_score": float(raw_data.get("notebook_quality_score", 0.0)),
                "active_last_year": bool(raw_data.get("active_last_year", False)),
                "strongest_domain": raw_data.get("strongest_domain", ""),
            }
            return cleaned
        except (TypeError, ValueError) as e:
            log.error("kaggle_data_cleaning_failed", error=str(e))
            return {"error": "Failed to clean Kaggle data"}

    def extract_document_text(self, path: Path) -> str:
        """Route to the correct extractor and enforce token budget."""
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                text = self._extract_text_from_pdf(path)
            elif suffix in (".md", ".markdown"):
                text = self._extract_text_from_markdown(path)
            elif suffix == ".txt":
                text = path.read_text(encoding="utf-8")
            else:
                log.warning("unsupported_file_type", suffix=suffix)
                return ""
            
            # Simple sanitization
            text = self.sanitize_text(text)
            
            # Enforce budget (heuristic: 1 token ~= 4 chars)
            char_limit = self.token_budget * 4
            if len(text) > char_limit:
                log.info("token_budget_exceeded", path=str(path), original_len=len(text))
                text = text[:char_limit] + "\n... [Content Truncated due to Token Budget] ..."
            
            return text
        except Exception as e:
            log.error("document_extraction_failed", path=str(path), error=str(e))
            return f"Error extracting text from {path.name}"

    def _extract_text_from_pdf(self, path: Path) -> str:
        text = []
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                if page_text := page.extract_text():
                    text.append(page_text)
        return "\n".join(text)

    def _extract_text_from_markdown(self, path: Path) -> str:
        content = path.read_text(encoding="utf-8")
        html = self._md.render(content)
        # Better tag stripping than raw regex
        text = re.sub(r"<[^>]+>", "", html)
        return text

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Remove excessive whitespace and non-printable characters."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x20-\x7E\n]", "", text)  # Keep printable ASCII and newlines
        return text.strip()

    def deduplicate_session_notes(self, notes: List[str], threshold: float = 0.85) -> List[str]:
        """Remove near-duplicate notes using Jaccard similarity."""
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

# --- Legacy Functional Wrappers for Backward Compatibility ---

_instance = DataPreprocessor()

def clean_github_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    return _instance.clean_github_data(raw_data)

def clean_kaggle_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    return _instance.clean_kaggle_data(raw_data)

def extract_document_text(path: Path) -> str:
    return _instance.extract_document_text(path)

def deduplicate_session_notes(notes: List[str], threshold: float = 0.85) -> List[str]:
    return _instance.deduplicate_session_notes(notes, threshold)

def hash_document_content(content: str) -> str:
    """Generate SHA-256 hash of document content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()