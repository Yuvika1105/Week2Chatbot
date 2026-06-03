"""RBAC manager for role-based file access and dynamic system prompts."""
from __future__ import annotations
from pathlib import Path
from typing import List

from config import Config


class RBACManager:
    """Simple RBAC manager mapping user ids to roles and permitted KB folders.

    User ids (strings) map to role names in `USER_DIRECTORY`. Role names map
    to permitted directories in `ROLE_PERMISSIONS`.
    """

    ROLE_PERMISSIONS = {
        "admin": ["hr_files", "finance_files", "it_files", "public"],
        "hr_user": ["hr_files", "public"],
        "finance_user": ["finance_files", "public"],
        "it_user": ["it_files", "public"],
        "guest": ["public"],
    }

    USER_DIRECTORY = {
        "1": {"role": "admin"},
        "2": {"role": "hr_user"},
        "3": {"role": "finance_user"},
        "4": {"role": "it_user"},
        "5": {"role": "guest"},
    }

    allowed_documents = {
        "admin": ["hr", "finance", "it", "public", "mg_data"],
        "hr_user": ["hr", "public"],
        "finance_user": ["finance", "public", "mg_data"],
        "it_user": ["it", "public", "mg_data"],
        "guest": ["public"],
    }

    def _normalize_uid(self, user_id: str) -> str:
        if isinstance(user_id, str) and user_id.startswith("u00"):
            return user_id[-1]
        return str(user_id)

    def get_allowed_documents(self, user_id: str) -> list[str]:
        user_id = self._normalize_uid(user_id)
        role = self.get_role(user_id)
        if role is None:
            return []
        return self.allowed_documents.get(role, [])

    def get_role(self, user_id: str) -> str | None:
        user_id = self._normalize_uid(user_id)
        rec = self.USER_DIRECTORY.get(user_id)
        return rec.get("role") if rec else None

    def get_user(self, user_id: str) -> dict | None:
        user_id = self._normalize_uid(user_id)
        return self.USER_DIRECTORY.get(user_id)

    def get_rag_document_groups(self, user_id: str) -> list:
        """Return logical RAG groups (short names) for the user's role.

        Tests expect admin to include 'all' and guests to return an empty list.
        """
        user_id = self._normalize_uid(user_id)
        role = self.get_role(user_id)
        if role is None:
            return []
        if role == "admin":
            return ["all"]
        mapping = {
            "hr_user": ["hr"],
            "finance_user": ["finance"],
            "it_user": ["it"],
            "guest": [],
        }
        return mapping.get(role, [])

    def get_sql_tables(self, user_id: str) -> list:
        """Return list of SQL table names the role is permitted to query (test-only convenience)."""
        user_id = self._normalize_uid(user_id)
        role = self.get_role(user_id)
        if role is None:
            return []
        mapping = {
            "admin": ["employees", "payroll", "it_assets"],
            "hr_user": ["employees"],
            "finance_user": ["payroll"],
            "it_user": ["it_assets"],
            "guest": [],
        }
        return mapping.get(role, [])

    def can_use_chatbot(self, user_id: str) -> bool:
        user_id = self._normalize_uid(user_id)
        return self.get_role(user_id) is not None

    def get_allowed_files(self, user_id: str, base_kb_dir: str | None = None) -> List[str]:
        """Return absolute paths to .txt files within permitted folders.

        If the user is unknown, return an empty list (fail-closed).
        """
        user_id = self._normalize_uid(user_id)
        role = self.get_role(user_id)
        if role is None:
            return []

        base_dir = base_kb_dir or Config.BASE_KB_DIR
        base = Path(base_dir)
        allowed_dirs = self.ROLE_PERMISSIONS.get(role, [])
        files: List[str] = []

        for d in allowed_dirs:
            p = base / d
            if not p.exists() or not p.is_dir():
                continue
            for f in sorted(p.glob("*.txt")):
                try:
                    files.append(str(f.resolve()))
                except Exception:
                    files.append(str(f))

        return files

    def get_system_prompt(self, user_id: str) -> str:
        """Return a role-aware system prompt based on the user's role.

        Unknown users get a conservative guest prompt (fail-closed).
        """
        user_id = self._normalize_uid(user_id)
        role = self.get_role(user_id)
        if role == "admin":
            return (
                "You are a secure administrative assistant. You may access internal admin documents. "
                "Answer only when the request relates to documents you are permitted to access. "
                "If the user asks about documents outside your permission, refuse and say you cannot access them."
            )
        if role == "hr_user":
            return (
                "You are an HR assistant. Answer HR policy and employee-related questions. "
                "Do not provide financial or IT details. If a request requires documents you don't have access to, state you cannot access them."
            )
        if role == "finance_user":
            return (
                "You are a Finance assistant. Answer finance and expense-related questions. "
                "Do not provide HR or IT details. If a request requires documents you don't have access to, state you cannot access them."
            )
        if role == "it_user":
            return (
                "You are an IT assistant. Answer IT support and security questions. "
                "Do not provide HR or Finance details. If a request requires documents you don't have access to, state you cannot access them."
            )

        # Default guest/unknown
        return (
            "You are a public FAQ assistant. Provide only general, non-sensitive information. "
            "If the user asks for internal or privileged documents, state that you do not have access."
        )


# Provide a backward-compatible name expected by the rest of the codebase
RBAC = RBACManager()
