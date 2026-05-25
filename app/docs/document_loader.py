from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load plain-text documents from a `documents/<group>/` folder.

    Each role maps to one or more document groups (see `RBAC`). This loader
    returns concatenated text for the groups the user is allowed to access.
    """

    def __init__(self, base_path: str | None = None) -> None:
        # Determine repository root relative to this file when not provided
        if base_path:
            self.base = Path(base_path)
        else:
            # file -> docs -> app -> <repo>
            self.base = Path(__file__).resolve().parents[2] / "documents"

        # Ensure folder exists
        try:
            os.makedirs(self.base, exist_ok=True)
        except Exception:
            pass

    def load_docs_for_groups(self, groups: Iterable[str]) -> str:
        """Return concatenated document contents for the provided groups.

        Parameters
        ----------
        groups : iterable of str
            Directory names under `documents/` to read. Files are read in
            alphabetical order; non-text files are skipped.
        """
        collected = []

        for group in groups:
            group_dir = self.base / group
            if not group_dir.exists() or not group_dir.is_dir():
                continue

            for path in sorted(group_dir.glob("*.txt")):
                try:
                    text = path.read_text(encoding="utf-8")
                except Exception as exc:
                    logger.debug("Skipping %s: %s", path, exc)
                    continue

                header = f"--- DOCUMENT: {group}/{path.name} ---\n"
                collected.append(header + text + "\n\n")

        return "".join(collected)
