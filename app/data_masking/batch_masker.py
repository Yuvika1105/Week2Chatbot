import argparse
import json
from pathlib import Path
from app.data_masking.masking_policy import MaskingPolicy
from app.data_masking.masking_engine import MaskingEngine
from app.data_masking.file_processors import FileProcessor

def main():
    parser = argparse.ArgumentParser(description="Batch Masker CLI")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input), Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = MaskingEngine(policy=MaskingPolicy.from_yaml(args.policy))
    report = {"files_processed": 0}

    for file_path in input_dir.iterdir():
        if file_path.is_dir() or file_path.name.startswith("."): continue
        ext = file_path.suffix.lower()
        try:
            if ext == ".txt": content = FileProcessor.process_txt(file_path, engine)
            elif ext == ".pdf": content = FileProcessor.process_pdf(file_path, engine)
            elif ext == ".docx": content = FileProcessor.process_docx(file_path, engine)
            elif ext == ".csv": content = FileProcessor.process_csv(file_path, engine)
            elif ext in [".xls", ".xlsx"]:
                content = FileProcessor.process_excel(file_path, engine)
                ext = ".csv"
            elif ext == ".json": content = FileProcessor.process_json(file_path, engine)
            else: continue
            
            out_name = f"{file_path.stem}_masked{ext}"
            with open(output_dir / out_name, "w", encoding="utf-8") as out_f:
                out_f.write(content)
            report["files_processed"] += 1
            print(f"  [OK] {file_path.name} -> {out_name}")
        except Exception as e:
            print(f"  [ERROR] {file_path.name}: {e}")

    with open(output_dir / "masking_report.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()