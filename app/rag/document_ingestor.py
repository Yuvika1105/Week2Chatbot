from pathlib import Path
from app.data_masking.masking_policy import MaskingPolicy
from app.data_masking.masking_engine import MaskingEngine
from app.data_masking.file_processors import FileProcessor

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