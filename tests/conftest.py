import os
import pytest
from config import Config

@pytest.fixture(autouse=True)
def use_test_db(tmp_path):
    """Automatically redirect the memory DB to a temp directory for all tests to ensure isolation."""
    test_db = tmp_path / "test_chatbot_memory.db"
    old_path = Config.MEMORY_DB_PATH
    Config.MEMORY_DB_PATH = str(test_db)
    yield
    Config.MEMORY_DB_PATH = old_path
