"""ChromaDB RAG store."""

 # src/memory/vector_store.py

from typing import Optional, List, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
from src.state.schema import UserProfile


class VectorStore:
    """Manages long-term memory using ChromaDB + SentenceTransformers."""
    
    def __init__(self, db_path="./data/vector_store"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self.client.get_or_create_collection("coach_memory")
        
    def upsert(self, 
               user_id: str,
               text: str,
               metadata: dict, 
               doc_id: Optional[str] = None) -> str:
        """Add or update a document."""
        if doc_id is None:
            doc_id = f"{user_id}:{text[:20]}"
        
        embedding = self.model.encode([text], convert_to_numpy=True).tolist()[0]
        
        self.collection.upsert(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        return doc_id
    
    def query(self, 
              user_id: str,
              query_text: str, 
              top_k: int = 5) -> List[dict]:
        """Search for similar documents."""
        embedding = self.model.encode([query_text], convert_to_numpy=True).tolist()[0]
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where={"user_id": user_id}
        )
        
        # Format results
        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": results["distances"][0][i]
            })
        return docs
    
    def get_by_type(self, user_id: str, doc_type: str) -> List[dict]:
        """Get documents of a specific type."""
        results = self.collection.get(
            where={"user_id": user_id, "type": doc_type}
        )
        
        docs = []
        for i in range(len(results["ids"])):
            docs.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return docs


class MemoryManager:
    """Helper for memory-related operations."""
    
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
    
    def persist_session_summary(self, 
                                user_id: str,
                                session_id: str,
                                learning_plan: dict,
                                user_profile: UserProfile) -> str:
        """Create and persist a session summary."""
        summary = f"""
        Session Summary - {session_id}
        
        Learning Plan:
        - Current week: {learning_plan['current_week']} of {len(learning_plan['weeks'])}
        - Target role: {user_profile.target_role}
        - Skills: {', '.join(user_profile.current_skills)}
        
        Notes from this session...
        """  # (in real implementation, pull from session notes)
        
        return self.vs.upsert(
            user_id=user_id,
            text=summary,
            metadata={
                "type": "session_summary",
                "session_id": session_id,
                "week": learning_plan['current_week']
            }
        )
    
    def retrieve_relevant_context(self, user_id: str, user_query: str) -> List[str]:
        """Retrieve relevant context from memory."""
        # Retrieve similar past sessions
        sessions = self.vs.query(user_id, user_query, top_k=3)
        
        # Retrieve user's learning plan
        plan_docs = self.vs.get_by_type(user_id, "learning_plan")
        
        # Retrieve profile summary
        profile_docs = self.vs.get_by_type(user_id, "profile_summary")
        
        contexts = []
        contexts.extend([s["text"] for s in sessions])
        contexts.extend([p["text"] for p in plan_docs])
        contexts.extend([p["text"] for p in profile_docs])
        
        return contexts