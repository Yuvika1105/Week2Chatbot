import os
import csv
import io
from pathlib import Path
from langchain_core.documents import Document
from app.rag.vector_store import LocalVectorStore
from app.rag.document_ingestor import DocumentIngestor

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """Split text into overlapping chunks of a given character size."""
    paragraphs = text.split("\n\n")
    chunks = []
    for p in paragraphs:
        p = p.strip()
        if not p: continue
        if len(p) <= chunk_size:
            chunks.append(p)
        else:
            # Simple character sliding window
            start = 0
            while start < len(p):
                chunks.append(p[start:start+chunk_size])
                start += (chunk_size - chunk_overlap)
    return chunks

def main():
    print("=" * 60)
    print("   SECURE GENAI PIPELINE SETUP & INGESTION")
    print("=" * 60)

    # Initialize Vector Store
    vector_store = LocalVectorStore()
    vector_store.clear()
    print("[-] Cleared existing vector store database.")

    # Initialize Ingestors
    default_ingestor = DocumentIngestor("data/masking_policies/default_policy.yaml")
    mg_ingestor = DocumentIngestor("data/masking_policies/mg_policy.yaml")

    total_chunks = 0

    # 1. Ingest standard KB documents
    kb_dirs = {
        "hr": "knowledge_base/hr_files",
        "finance": "knowledge_base/finance_files",
        "it": "knowledge_base/it_files",
        "public": "knowledge_base/public"
    }

    for category, dir_path in kb_dirs.items():
        p = Path(dir_path)
        if not p.exists(): continue
        for file_path in p.glob("*.txt"):
            print(f"[+] Ingesting standard file: {file_path.name} (category: {category})")
            masked_content = default_ingestor.prepare_file_content(str(file_path))
            chunks = chunk_text(masked_content)
            
            docs = [
                Document(
                    page_content=chunk,
                    metadata={"document_category": category, "source": file_path.name}
                )
                for chunk in chunks
            ]
            vector_store.add_documents(docs)
            total_chunks += len(docs)

    # 2. Ingest MG Motors documents
    mg_dir = Path("mg_data_masking")
    if mg_dir.exists():
        for file_path in mg_dir.glob("*.csv"):
            print(f"[+] Ingesting MG Motors raw file with pre-masking: {file_path.name}")
            masked_csv_str = mg_ingestor.prepare_file_content(str(file_path))
            
            # Read rows from the masked CSV string
            f_in = io.StringIO(masked_csv_str)
            reader = csv.DictReader(f_in)
            
            docs = []
            for idx, row in enumerate(reader):
                row_str = ", ".join([f"{k}: {v}" for k, v in row.items() if v])
                docs.append(
                    Document(
                        page_content=row_str,
                        metadata={"document_category": "mg_data", "source": f"{file_path.name} [Row {idx+1}]"}
                    )
                )
            
            vector_store.add_documents(docs)
            total_chunks += len(docs)

    print("-" * 60)
    print(f"[OK] Ingestion completed. Total chunks loaded: {total_chunks}")
    print("=" * 60)

if __name__ == "__main__":
    main()
