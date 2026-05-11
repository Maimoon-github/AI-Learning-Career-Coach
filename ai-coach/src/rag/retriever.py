"""Adaptive retriever."""

# src/rag/retriever.py

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import os


class AgenticRetriever:
    """
    Agentic RAG retriever with query rewriting and relevance checking.
    Implements: Query → Retrieve → Grade → Rewrite (if needed) → Return
    """

    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=os.environ["EMBEDDING_MODEL"])
        self.vectorstore = Chroma(
            persist_directory=os.environ["CHROMA_PERSIST_DIR"],
            embedding_function=self.embeddings,
            collection_name="course_materials",
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",              # Maximum Marginal Relevance for diversity
            search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7}
        )
        self.grader_llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M", temperature=0)

    def retrieve(self, query: str, user_context: str = "", max_rewrite_attempts: int = 2) -> list[dict]:
        """
        Full agentic retrieval cycle with relevance grading.
        """
        docs = self.retriever.invoke(query)
        graded = self._grade_documents(query, docs)

        if len(graded) < 2 and max_rewrite_attempts > 0:
            rewritten_query = self._rewrite_query(query, user_context)
            return self.retrieve(rewritten_query, user_context, max_rewrite_attempts - 1)

        return [{"content": d.page_content, "metadata": d.metadata, "source": d.metadata.get("source", "")}
                for d in graded]

    def _grade_documents(self, query: str, docs) -> list:
        """Grade each retrieved document for relevance."""
        grader_prompt = ChatPromptTemplate.from_messages([
            ("human",
             "Is this document relevant to the query?\n"
             "Query: {query}\nDocument: {doc}\n"
             "Answer with only 'yes' or 'no'.")
        ])
        chain = grader_prompt | self.grader_llm
        return [
            doc for doc in docs
            if "yes" in chain.invoke({"query": query, "doc": doc.page_content}).content.lower()
        ]

    def _rewrite_query(self, original: str, context: str) -> str:
        """Rewrite the query to be more specific/effective."""
        prompt = ChatPromptTemplate.from_messages([
            ("human",
             "The following query returned poor retrieval results. "
             "Rewrite it to be more specific and likely to find relevant educational content.\n"
             "Original query: {query}\nUser context: {context}\nRewritten query:")
        ])
        chain = prompt | self.grader_llm
        result = chain.invoke({"query": original, "context": context})
        return result.content.strip()