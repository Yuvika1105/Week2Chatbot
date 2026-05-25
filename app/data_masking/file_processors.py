import json
import pandas as pd
import pdfplumber
from docx import Document
from pathlib import Path
from app.data_masking.masking_engine import MaskingEngine

class FileProcessor:
    @staticmethod
    def process_txt(path: Path, engine: MaskingEngine) -> str:
        return engine.mask_text(path.read_text(encoding="utf-8")).masked_text

    @staticmethod
    def process_pdf(path: Path, engine: MaskingEngine) -> str:
        extracted = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text: extracted.append(text)
        return engine.mask_text("\n".join(extracted)).masked_text

    @staticmethod
    def process_docx(path: Path, engine: MaskingEngine) -> str:
        doc = Document(path)
        full_text = "\n".join([p.text for p in doc.paragraphs])
        return engine.mask_text(full_text).masked_text

    @staticmethod
    def process_csv(path: Path, engine: MaskingEngine) -> str:
        df = pd.read_csv(path)
        for col in df.columns:
            if col in engine.policy.column_rules:
                df[col] = df[col].astype(str).apply(lambda v: engine.mask_value(col, v))
            else:
                df[col] = df[col].astype(str).apply(lambda v: engine.mask_text(v).masked_text)
        return df.to_csv(index=False)

    @staticmethod
    def process_excel(path: Path, engine: MaskingEngine) -> str:
        df = pd.read_excel(path)
        for col in df.columns:
            if col in engine.policy.column_rules:
                df[col] = df[col].astype(str).apply(lambda v: engine.mask_value(col, v))
            else:
                df[col] = df[col].astype(str).apply(lambda v: engine.mask_text(v).masked_text)
        return df.to_csv(index=False)

    @staticmethod
    def process_json(path: Path, engine: MaskingEngine) -> str:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return engine.mask_text(json.dumps(data)).masked_text