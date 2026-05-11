"""Document ingestion pipeline."""

# src/rag/ingestion.py

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os


def ingest_documents(file_paths: list[str], collection_name: str = "course_materials") -> int:
    """
    Ingest documents into ChromaDB. Returns number of chunks stored.
    Supports: PDF, Markdown, TXT
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    all_docs = []
    for path in file_paths:
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(path)
        else:
            loader = TextLoader(path)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)
        # Tag with source
        for split in splits:
            split.metadata["source"] = path
        all_docs.extend(splits)

    embeddings = OllamaEmbeddings(model=os.environ["EMBEDDING_MODEL"])
    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=os.environ["CHROMA_PERSIST_DIR"],
        collection_name=collection_name,
    )
    return len(all_docs)