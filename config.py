# config.py
import os
from dotenv import load_dotenv

# Load variables locally if .env exists
load_dotenv()

class Config:
    # Safely look for environment variables
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MAIN_MODEL = os.getenv("MAIN_MODEL", "llama-3.3-70b-versatile")
    PROMPT_GUARD_MODEL = os.getenv("PROMPT_GUARD_MODEL", "meta-llama/llama-prompt-guard-2-86m")
    SAFEGUARD_MODEL = os.getenv("SAFEGUARD_MODEL", "openai/gpt-oss-safeguard-20b")

    # Data and storage defaults
    BASE_KB_DIR = os.getenv("BASE_KB_DIR", "knowledge_base")
    AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", os.path.join("logs", "audit.jsonl"))
    MG_DOCS_DIR = os.getenv("MG_DOCS_DIR", "mg_data_masking")
    MASKING_POLICIES_DIR = os.getenv("MASKING_POLICIES_DIR", os.path.join("data", "masking_policies"))

# Set environment variables from Config for guardrail reusability
# This allows guardrails to work with environment variables while maintaining backward compatibility
os.environ.setdefault("GROQ_API_KEY", Config.GROQ_API_KEY or "")
os.environ.setdefault("PROMPT_GUARD_MODEL", Config.PROMPT_GUARD_MODEL)
os.environ.setdefault("SAFEGUARD_MODEL", Config.SAFEGUARD_MODEL)

# Enforce logging storage path structure directory creation dynamically
os.makedirs("logs", exist_ok=True)