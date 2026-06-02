import os
import json
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document

class LocalVectorStore:
    def __init__(self, db_path: str = "data/vector_store_db.json"):
        self.db_path = db_path
        self.documents: List[Document] = []
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents = [
                    Document(page_content=item["page_content"], metadata=item["metadata"])
                    for item in data
                ]
            except Exception:
                self.documents = []
        else:
            self.documents = []

    def save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in self.documents
        ]
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_documents(self, documents: List[Document]):
        # Prevent duplicate document paths or contents if needed
        self.documents.extend(documents)
        self.save()

    def clear(self):
        self.documents = []
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def similarity_search(self, query: str, k: int = 4, filter: Dict[str, Any] = None) -> List[Document]:
        # Filter documents first
        filtered_docs = []
        for doc in self.documents:
            if filter:
                match = True
                for fk, fv in filter.items():
                    val = doc.metadata.get(fk)
                    if isinstance(fv, list):
                        if "all" in fv:
                            continue
                        if val not in fv:
                            match = False
                            break
                    else:
                        if fv == "all":
                            continue
                        if val != fv:
                            match = False
                            break
                if not match:
                    continue
            filtered_docs.append(doc)

        # Score documents
        scored_docs: List[Tuple[float, Document]] = []
        query_lower = query.lower()
        import re
        clean_query = re.sub(r'[^\w\s]', '', query_lower)
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "to", "from", "in", "on", "at", "by", "for", "with", "about", "of", "it", "its", "they", "them", "their", "he", "him", "his", "she", "her", "you", "your", "we", "us", "our", "what", "which", "who", "whom", "this", "that", "these", "those"}
        query_words = [w for w in clean_query.split() if len(w) >= 2 and w not in stop_words]
        if not query_words:
            query_words = [w for w in clean_query.split() if len(w) >= 2]

        for doc in filtered_docs:
            content_lower = doc.page_content.lower()
            score = 0.0
            
            # Phrase match booster
            if query_lower in content_lower:
                score += 10.0
            
            # Individual word booster
            for word in query_words:
                if word in content_lower:
                    score += 1.0

            scored_docs.append((score, doc))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:k]]
