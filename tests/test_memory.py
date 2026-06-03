import os
import sqlite3
import pytest
from config import Config
from app.context_engineering.context_manager import ConversationContextManager
from app.chatbot.chatbot import SecuredChatbot
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
import json

def _make_groq_llm_mock(response_text: str) -> MagicMock:
    mock_llm = MagicMock()
    mock_message = MagicMock()
    mock_message.content = response_text
    mock_llm.invoke.return_value = mock_message
    return mock_llm

def _clean_safeguard_mock(mocker):
    clean = json.dumps({"violation": 0, "category": "", "severity": "", "rationale": "Safe."})
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = clean
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.return_value = mock_resp
    mocker.patch("app.guardrails.toxicity_checker.Groq", return_value=mock_groq)
    mocker.patch("app.guardrails.input_guardrail.Groq", return_value=mock_groq)

class TestSQLiteMemory:

    def test_add_and_retrieve_turns(self):
        mgr = ConversationContextManager(user_id="user_1")
        
        # Initially empty
        assert len(mgr.get_turns()) == 0
        
        # Add turn
        mgr.add_turn("Hello bot", "Hello human")
        turns = mgr.get_turns()
        assert len(turns) == 1
        assert turns[0]["user"] == "Hello bot"
        assert turns[0]["ai"] == "Hello human"
        
        # Add another turn
        mgr.add_turn("How are you?", "I am good")
        turns = mgr.get_turns()
        assert len(turns) == 2
        assert turns[1]["user"] == "How are you?"
        assert turns[1]["ai"] == "I am good"

    def test_user_isolation(self):
        mgr_1 = ConversationContextManager(user_id="user_1")
        mgr_2 = ConversationContextManager(user_id="user_2")
        
        mgr_1.add_turn("User 1 query", "User 1 response")
        mgr_2.add_turn("User 2 query", "User 2 response")
        
        turns_1 = mgr_1.get_turns()
        turns_2 = mgr_2.get_turns()
        
        assert len(turns_1) == 1
        assert turns_1[0]["user"] == "User 1 query"
        
        assert len(turns_2) == 1
        assert turns_2[0]["user"] == "User 2 query"

    def test_sliding_window_token_limits(self):
        # The window limits by default: 800 tokens * 4 = 3200 characters.
        # Let's write extremely large messages to trigger the window limit.
        mgr = ConversationContextManager(user_id="user_1")
        
        # Each turn is roughly 1000 chars.
        large_msg_1 = "A" * 1000
        large_msg_2 = "B" * 1000
        large_msg_3 = "C" * 1000
        large_msg_4 = "D" * 1000
        
        mgr.add_turn(large_msg_1, large_msg_1) # ~2000 chars total
        mgr.add_turn(large_msg_2, large_msg_2) # ~2000 chars total
        mgr.add_turn(large_msg_3, large_msg_3) # ~2000 chars total
        mgr.add_turn(large_msg_4, large_msg_4) # ~2000 chars total
        
        # Total turns in database is 4.
        assert len(mgr.get_turns()) == 4
        
        # get_window(max_tokens=800) should only return the last 2 turns, 
        # since last 2 turns take ~4000 characters, exceeding 3200 characters if we added the third turn.
        # Wait, let's verify character count:
        # Turn 4: User: 1000 chars + AI: 1000 chars + format = 2013 chars
        # Turn 3: User: 1000 chars + AI: 1000 chars + format = 2013 chars
        # Sum = 4026 chars, which is > 3200 chars. So only Turn 4 (the latest) will fit because Turn 4 + Turn 3 exceeds max_chars!
        window = mgr.get_window(max_tokens=800)
        assert len(window) == 1
        assert window[0]["user"] == large_msg_4

    def test_clear_history(self):
        mgr_1 = ConversationContextManager(user_id="user_1")
        mgr_2 = ConversationContextManager(user_id="user_2")
        
        mgr_1.add_turn("Q1", "A1")
        mgr_2.add_turn("Q2", "A2")
        
        # Clear user_1
        mgr_1.clear()
        
        assert len(mgr_1.get_turns()) == 0
        # user_2 remains unaffected!
        assert len(mgr_2.get_turns()) == 1
        assert mgr_2.get_turns()[0]["user"] == "Q2"

    def test_chatbot_integration_injects_history(self, mocker):
        _clean_safeguard_mock(mocker)
        bot = SecuredChatbot()
        bot._llm = _make_groq_llm_mock("I remember you.")
        
        # Add history turn directly
        bot.history.user_id = "1"
        bot.history.add_turn("My name is Alice", "Nice to meet you, Alice")
        
        # Call process_message
        res = bot.process_message("1", "What is my name?")
        
        # Assert history was sent to LLM
        called_messages = bot._llm.invoke.call_args[0][0]
        # Messages structure: [SystemMessage(combined_prompt), HumanMessage(prev_user), AIMessage(prev_ai), HumanMessage(current_msg)]
        assert len(called_messages) >= 4
        assert any(isinstance(msg, AIMessage) and "Alice" in msg.content for msg in called_messages)
        assert called_messages[-1].content == "What is my name?"
        assert res == "I remember you."
        
        # Verify new turn was saved in DB
        turns = bot.history.get_turns()
        assert len(turns) == 2
        assert turns[1]["user"] == "What is my name?"
        assert turns[1]["ai"] == "I remember you."
