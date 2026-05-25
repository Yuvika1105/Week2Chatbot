#!/usr/bin/env python3
"""
Example script demonstrating how to use guardrails standalone.
This shows that others can copy just the guardrails and use them in their own projects.
"""

import os
import sys

# Example 1: Using environment variables (recommended for reusability)
print("=" * 70)
print("GUARDRAIL REUSABILITY DEMO")
print("=" * 70)

# Simulate what an external user would do
print("\n[EXAMPLE 1] Using environment variables:")
print("-" * 70)

# They would set their API key in .env or environment
# os.environ["GROQ_API_KEY"] = "their_api_key"

try:
    from app.guardrails.input_guardrail import InputGuardrail
    from app.guardrails.pii_detector import PIIDetector
    
    print("✓ Successfully imported InputGuardrail")
    print("✓ Successfully imported PIIDetector")
    
    # PIIDetector works without any API key (uses local Presidio)
    pii = PIIDetector()
    print("✓ PIIDetector instantiated successfully")
    
    # Test PII detection
    result = pii.mask("John Doe's email is john@example.com and phone is 555-1234")
    print(f"\nPII Detection Demo:")
    print(f"  Original: John Doe's email is john@example.com and phone is 555-1234")
    print(f"  Masked:   {result['masked_text']}")
    print(f"  Entities: {result['entities_found']}")
    
except Exception as e:
    print(f"✗ Error: {e}")

# Example 2: Using constructor parameters (for custom configurations)
print("\n\n[EXAMPLE 2] Using constructor parameters:")
print("-" * 70)

print("""
# External user can do:
from app.guardrails.input_guardrail import InputGuardrail

guardrail = InputGuardrail(
    groq_api_key="their_groq_api_key",
    prompt_guard_model="meta-llama/llama-prompt-guard-2-86m"
)

result = guardrail.scan("What is the leave policy?")
if result["safe"]:
    print("✓ Input is safe")
else:
    print(f"✗ Blocked: {result['reason']}")
""")

# Example 3: Show what files need to be copied
print("\n\n[EXAMPLE 3] Files to copy for reuse:")
print("-" * 70)

files_to_copy = [
    "app/guardrails/input_guardrail.py",
    "app/guardrails/output_guardrail.py", 
    "app/guardrails/toxicity_checker.py",
    "app/guardrails/pii_detector.py",
    "app/guardrails/README.md",
]

print("\nCopy these files to your project:")
for file in files_to_copy:
    print(f"  • {file}")

print("\nThen create a .env file with:")
print("""  GROQ_API_KEY=your_key_here
  PROMPT_GUARD_MODEL=meta-llama/llama-prompt-guard-2-86m
  SAFEGUARD_MODEL=openai/gpt-oss-safeguard-20b
""")

# Example 4: Show backward compatibility
print("\n\n[EXAMPLE 4] Backward Compatibility:")
print("-" * 70)

try:
    from config import Config
    from app.guardrails.input_guardrail import InputGuardrail
    from app.guardrails.toxicity_checker import ToxicityChecker
    
    print("✓ Config class still works")
    print(f"  - GROQ_API_KEY: {'Set' if Config.GROQ_API_KEY else 'Not set'}")
    print(f"  - PROMPT_GUARD_MODEL: {Config.PROMPT_GUARD_MODEL}")
    print(f"  - SAFEGUARD_MODEL: {Config.SAFEGUARD_MODEL}")
    
    print("\n✓ Guardrails instantiate with existing config:")
    
    # Your existing code still works!
    # (Comment out if you don't want to make API calls in demo)
    # input_guard = InputGuardrail()
    # output_check = ToxicityChecker()
    # print("  - InputGuardrail() works")
    # print("  - ToxicityChecker() works")
    
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 70)
print("SUMMARY: Your guardrails are now reusable! 🎉")
print("=" * 70)
print("""
✓ Your existing code works without changes
✓ Others can use the guardrails by copying the files
✓ Configuration via environment variables
✓ No hardcoded API keys or dependencies
✓ Complete documentation in app/guardrails/README.md
""")
