from pathlib import Path
import os
from app.data_masking.masking_policy import MaskingPolicy
from app.data_masking.masking_engine import MaskingEngine
from app.data_masking.file_processors import FileProcessor
from app.rag.vector_store import LocalVectorStore
from langchain_core.documents import Document

class DocumentIngestor:
    def __init__(self, policy_path: str):
        self.engine = MaskingEngine(policy=MaskingPolicy.from_yaml(policy_path))

    def prepare_file_content(self, file_path: str) -> str:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == ".txt": return FileProcessor.process_txt(path, self.engine)
        elif ext == ".csv": return FileProcessor.process_csv(path, self.engine)
        elif ext in [".xls", ".xlsx"]: return FileProcessor.process_excel(path, self.engine)
        elif ext == ".pdf": return FileProcessor.process_pdf(path, self.engine)
        elif ext == ".docx": return FileProcessor.process_docx(path, self.engine)
        return path.read_text(encoding="utf-8", errors="ignore")

def ingest_documents(clean_existing: bool = True):
    vector_store = LocalVectorStore()
    if clean_existing:
        vector_store.clear()

    # Scan directories
    kb_dirs = {
        "hr": "knowledge_base/hr_files",
        "finance": "knowledge_base/finance_files",
        "it": "knowledge_base/it_files",
        "public": "knowledge_base/public"
    }

    all_docs = []

    # Chunking helper
    def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
        paragraphs = text.split("\n\n")
        chunks = []
        for p in paragraphs:
            p = p.strip()
            if not p: continue
            if len(p) <= chunk_size:
                chunks.append(p)
            else:
                start = 0
                while start < len(p):
                    chunks.append(p[start:start+chunk_size])
                    start += (chunk_size - chunk_overlap)
        return chunks

    # Ingest standard KB documents
    for category, dir_path in kb_dirs.items():
        p = Path(dir_path)
        if not p.exists(): continue
        for file_path in p.glob("*.txt"):
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_text(content)
            for chunk in chunks:
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={"document_category": category, "source": file_path.name}
                ))

    # Ingest MG data documents
    mg_p = Path("mg_data_masking")
    if mg_p.exists():
        for file_path in mg_p.glob("*.csv"):
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_text(content)
            for chunk in chunks:
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={"document_category": "mg_data", "source": file_path.name}
                ))

    # Pre-ingestion masking block
    if all_docs:
        # Load default_policy.yaml and create MaskingEngine
        policy_path = os.path.join("data", "masking_policies", "default_policy.yaml")
        policy = MaskingPolicy.from_yaml(policy_path)
        masking_engine = MaskingEngine(policy=policy)

        for chunk in all_docs:
            result = masking_engine.mask_text(chunk.page_content)
            chunk.page_content = result.masked_text
            chunk.metadata["masking_applied"] = True

        vector_store.add_documents(all_docs)
        print(f"[OK] Ingested {len(all_docs)} masked chunks into the vector store.")