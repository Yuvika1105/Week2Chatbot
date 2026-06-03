import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import pytest

def main():
    parser = argparse.ArgumentParser(description="Secure GenAI Chatbot Test Suite Executor")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--module", choices=["rag", "guardrails", "all"], default="all", help="Test module to execute")
    args = parser.parse_args()

    pytest_args = []
    if args.verbose:
        pytest_args.append("-v")

    if args.module == "guardrails":
        pytest_args.append("tests/test_guardrails.py")
    elif args.module == "rag":
        # Run masking and data tests
        pytest_args.extend(["tests/test_masking.py", "tests/test_rag.py"])
        # We also include any rag integration tests
        # If there are others, we could append them here
    else:
        # Run all test files
        pytest_args.append("tests/")

    print(f"Executing pytest with arguments: {pytest_args}")
    result = pytest.main(pytest_args)
    sys.exit(result)

if __name__ == "__main__":
    main()
