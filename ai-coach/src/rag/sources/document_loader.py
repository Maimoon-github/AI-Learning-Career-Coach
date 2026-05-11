"""Generic document loader."""

# src\rag\sources\document_loader.py

from typing import List, Dict, Any
import os


class DocumentLoader:
    """Base class for document loaders."""
    
    def load(self, source: Any) -> List[Dict[str, Any]]:
        """Load documents from a source."""
        raise NotImplementedError


class FileLoader(DocumentLoader):
    """Load documents from files."""
    
    def load(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load documents from file paths."""
        documents = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        documents.append({
                            "text": text,
                            "source": file_path
                        })
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        return documents


class WebLoader(DocumentLoader):
    """Load documents from web URLs (basic implementation)."""
    
    def load(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Load documents from URLs."""
        documents = []
        for url in urls:
            try:
                # In production, use BeautifulSoup or Scrapy
                # For now, return a placeholder
                documents.append({
                    "text": f"Content from {url} would be scraped here.",
                    "source": url
                })
            except Exception as e:
                print(f"Error loading {url}: {e}")
        return documents


class YoutubeLoader(DocumentLoader):
    """Load transcripts from YouTube videos."""
    
    def load(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """Load YouTube video transcripts."""
        documents = []
        for video_id in video_ids:
            try:
                # This would use pytube or youtube-dl
                documents.append({
                    "text": f"Transcript from YouTube video {video_id} would be loaded here.",
                    "source": f"https://www.youtube.com/watch?v={video_id}"
                })
            except Exception as e:
                print(f"Error loading video {video_id}: {e}")
        return documents