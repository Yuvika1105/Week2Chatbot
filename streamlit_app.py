import io
import os
import tempfile
import uuid
from pathlib import Path
import streamlit as st

from app.data_masking.masking_engine import MaskingEngine
from app.data_masking.masking_policy import MaskingPolicy
from app.data_masking.file_processors import FileProcessor
from app.rag.secure_rag import SecureRAGPipeline
from app.chatbot.chatbot import SecuredChatbot
from app.auth.rbac import RBAC

DEFAULT_POLICY_PATH = "data/masking_policies/default_policy.yaml"

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list:
    """Split text into overlapping chunks of a given character size without cutting words in half."""
    paragraphs = text.split("\n\n")
    chunks = []
    for p in paragraphs:
        p = p.strip()
        if not p: continue
        if len(p) <= chunk_size:
            chunks.append(p)
        else:
            start = 0
            while start < len(p):
                end = start + chunk_size
                # If we are not at the end of the text, back up to the nearest space or newline
                if end < len(p):
                    last_space = p.rfind(" ", start, end)
                    last_newline = p.rfind("\n", start, end)
                    split_idx = max(last_space, last_newline)
                    if split_idx > start:
                        end = split_idx
                
                chunk = p[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                
                start = end - chunk_overlap
                # Prevent infinite loop if we can't advance
                if start <= end - chunk_size:
                    start = end
    return chunks

def process_chat_uploaded_file(uploaded_file, engine) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"chat_upload{suffix}"
        tmp_path.write_bytes(uploaded_file.getvalue())

        if suffix == ".txt":
            return FileProcessor.process_txt(tmp_path, engine)
        elif suffix == ".pdf":
            return FileProcessor.process_pdf(tmp_path, engine)
        elif suffix == ".docx":
            return FileProcessor.process_docx(tmp_path, engine)
        elif suffix == ".csv":
            return FileProcessor.process_csv(tmp_path, engine)
        elif suffix in [".xls", ".xlsx"]:
            return FileProcessor.process_excel(tmp_path, engine)
        elif suffix == ".json":
            return FileProcessor.process_json(tmp_path, engine)
        else:
            return engine.mask_text(uploaded_file.getvalue().decode("utf-8", errors="ignore")).masked_text


def _load_engine(policy_path: str) -> MaskingEngine:
    policy = MaskingPolicy.from_yaml(policy_path)
    return MaskingEngine(policy=policy)

@st.cache_resource(show_spinner=False)
def get_engine(policy_path: str) -> MaskingEngine:
    return _load_engine(policy_path)

@st.cache_resource(show_spinner=False)
def get_secure_rag_pipeline() -> SecureRAGPipeline:
    return SecureRAGPipeline()

@st.cache_resource(show_spinner=False)
def get_general_chatbot() -> SecuredChatbot:
    return SecuredChatbot()

st.set_page_config(page_title="Chat Bot", layout="wide", page_icon="🛡️")

# ── Global styles + fixed top header ─────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;800&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1.02rem;
    }

    /* ── Fixed top header bar ─────────────────────────────────── */
    /* Streamlit's own header bar is ~3.75rem tall.
       We pin our bar immediately below it, spanning the main pane.
       On Streamlit wide layout the sidebar is ~21rem wide. */
    #chat-header {
        position: fixed;
        top: 3.75rem;
        left: 21rem;
        right: 0;
        height: 3rem;
        z-index: 1000;
        background: #ffffff;
        border-bottom: 2px solid #e2e8f0;
        display: flex;
        align-items: center;
        padding: 0 1.5rem;
        gap: 0.6rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* When sidebar is collapsed Streamlit sets sidebar width to ~4rem */
    @media (max-width: 1024px) {
        #chat-header { left: 4rem; }
    }

    #chat-header .ch-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        color: #4f46e5;
        letter-spacing: -0.01em;
        margin: 0;
        padding: 0;
        line-height: 1;
    }

    #chat-header .ch-badge {
        display: inline-block;
        background: #ede9fe;
        color: #6d28d9;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 2px 10px;
        border-radius: 99px;
        white-space: nowrap;
        letter-spacing: 0.03em;
    }

    /* Push block-container below our fixed bar:
       3.75rem (Streamlit header) + 3rem (our bar) + 0.5rem gap = 7.25rem */
    .block-container {
        padding-top: 7.25rem !important;
        padding-bottom: 1rem !important;
        max-width: 860px;
    }

    /* Welcome card */
    .welcome-card {
        text-align: center;
        padding: 2.5rem 2rem;
        margin: 2rem auto;
        max-width: 440px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
    }
    .welcome-card .wc-heading {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #334155;
        margin: 0 0 0.35rem 0;
    }
    .welcome-card .wc-sub {
        font-size: 0.88rem;
        line-height: 1.65;
        color: #64748b;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# SIDEBAR NAVIGATION & SETTINGS (ChatGPT / Gemini style)
# =========================================================================

st.sidebar.markdown("### User Identity")
user_option = st.sidebar.selectbox(
    "Select User Identity",
    options=[
        "1 - Admin",
        "2 - HR User",
        "3 - Finance User",
        "4 - IT User",
        "5 - Guest"
    ]
)

user_mapping = {
    "1 - Admin": "1",
    "2 - HR User": "2",
    "3 - Finance User": "3",
    "4 - IT User": "4",
    "5 - Guest": "5"
}
user_id = user_mapping[user_option]
user_role = RBAC.get_role(user_id) or "guest"
st.sidebar.info(f"**Current Role**: `{user_role.upper()}`")

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
chat_mode = st.sidebar.radio(
    "Select Chat Mode",
    ["Secure RAG", "General Chatbot"]
)

# RAG Context Source Selection
context_source = "Standard Knowledge Base"
uploaded_chat_file = None
if chat_mode == "Secure RAG":
    context_source = st.sidebar.radio(
        "RAG Context Source",
        ["Standard Knowledge Base", "Uploaded Document"]
    )
    if context_source == "Uploaded Document":
        uploaded_chat_file = st.sidebar.file_uploader(
            "Upload Chat Context File",
            type=["pdf", "csv", "xlsx", "xls", "docx", "txt", "json"],
            key="chat_file_uploader"
        )
        if uploaded_chat_file is not None:
            st.sidebar.success(f"Loaded: `{uploaded_chat_file.name}`")

st.sidebar.markdown("---")
st.sidebar.markdown("### Chat History Logs")

# Initialize context manager
from app.context_engineering.context_manager import ConversationContextManager
ctx_mgr = ConversationContextManager(user_id=user_id)

# Fetch sessions
sessions = ctx_mgr.get_sessions()

# Automatically create a session if none exist
if not sessions:
    new_sid = str(uuid.uuid4())
    ctx_mgr.create_session(new_sid, "New Chat")
    sessions = ctx_mgr.get_sessions()

# Ensure active session_id is valid
if "session_id" not in st.session_state or st.session_state.session_id not in [s["session_id"] for s in sessions]:
    st.session_state.session_id = sessions[0]["session_id"]

active_sid = st.session_state.session_id

# "New Chat" Button
if st.sidebar.button("New Chat", use_container_width=True, type="primary"):
    new_sid = str(uuid.uuid4())
    existing_new_chats = [s for s in sessions if s["title"].startswith("New Chat")]
    new_num = len(existing_new_chats) + 1
    ctx_mgr.create_session(new_sid, f"New Chat {new_num}")
    st.session_state.session_id = new_sid
    st.session_state.chat_history = []
    st.session_state.rag_history = {}
    st.rerun()

st.sidebar.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

# Render session history buttons in sidebar (simple text, no emojis)
for s in sessions:
    col_btn, col_del = st.sidebar.columns([3, 1])
    
    label = s["title"]
    # Truncate label for sidebar if too long
    if len(label) > 20:
        label = label[:18] + "..."
        
    btn_label = f"Active: {label}" if s["session_id"] == active_sid else label
    btn_type = "primary" if s["session_id"] == active_sid else "secondary"
    
    if col_btn.button(btn_label, key=f"sel_{s['session_id']}", use_container_width=True, type=btn_type):
        st.session_state.session_id = s["session_id"]
        st.session_state.chat_history = []
        st.session_state.rag_history = {}
        st.rerun()
        
    if col_del.button("Delete", key=f"del_{s['session_id']}", help="Delete chat log"):
        ctx_mgr.delete_session(s["session_id"])
        if st.session_state.session_id == s["session_id"]:
            st.session_state.pop("session_id", None)
            st.session_state.pop("loaded_session_id", None)
        st.rerun()

st.sidebar.markdown("---")
clear_btn = st.sidebar.button("Clear Active Chat History", type="secondary", use_container_width=True)
if clear_btn:
    ctx_mgr.clear(session_id=active_sid)
    st.session_state.chat_history = []
    st.session_state.rag_history = {}
    st.rerun()

# =========================================================================
# MAIN CHAT TERMINAL
# =========================================================================

# Fixed header bar — always visible at top, never scrolls away
st.markdown(
    '<div id="chat-header"><span class="ch-title">Chat Bot</span></div>',
    unsafe_allow_html=True
)

# Reactively load history from SQLite when active session/user changes
if "loaded_session_id" not in st.session_state or st.session_state.loaded_session_id != active_sid or "current_user_id" not in st.session_state or st.session_state.current_user_id != user_id:
    st.session_state.loaded_session_id = active_sid
    st.session_state.current_user_id = user_id
    ctx_mgr.session_id = active_sid
    turns = ctx_mgr.get_turns()
    st.session_state.chat_history = []
    for turn in turns:
        st.session_state.chat_history.append({"role": "user", "content": turn["user"]})
        st.session_state.chat_history.append({"role": "assistant", "content": turn["ai"]})
    st.session_state.rag_history = {}

# Show welcome card when there are no messages yet
if not st.session_state.get("chat_history"):
    st.markdown(
        '<div class="welcome-card">'
        '<div class="wc-heading">How can I help you today?</div>'
        '<p class="wc-sub">Ask anything. I will answer based on the knowledge base<br>and keep your data secure throughout.</p>'
        '</div>',
        unsafe_allow_html=True
    )

# Render conversation history
for idx, chat in enumerate(st.session_state.chat_history):
    with st.chat_message(chat["role"]):
        st.write(chat["content"])
        
        # Render metadata/security logs for assistant messages
        if chat["role"] == "assistant" and idx in st.session_state.rag_history:
            rag_data = st.session_state.rag_history[idx]
            
            # Simple inspection
            if user_role == "admin":
                with st.expander("View Security Details"):
                    st.markdown(f"**Original Query**: `{rag_data['original_query']}`")
                    st.markdown(f"**Masked Query**: `{rag_data['masked_query']}`")
                    st.markdown(f"**Blocked**: `{rag_data['blocked']}`")
                    st.markdown(f"**Event ID**: `{rag_data['event_id']}`")
                    if rag_data.get("sources"):
                        st.markdown(f"**Sources**: `{', '.join(rag_data['sources'])}`")

# Chat Input
query = st.chat_input("Ask a question...")

if query:
    # 1. Display User Message
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # 2. Process Bot Response
    with st.chat_message("assistant"):
        with st.spinner("Processing securely..."):
            if chat_mode == "Secure RAG":
                if context_source == "Uploaded Document" and uploaded_chat_file is None:
                    response_text = "Please upload a document in the Settings sidebar to chat about it."
                    st.write(response_text)
                    res = {
                        "response": response_text,
                        "sources": [],
                        "masked_query": query,
                        "blocked": False,
                        "event_id": ""
                    }
                else:
                    custom_docs = None
                    if context_source == "Uploaded Document" and uploaded_chat_file is not None:
                        engine = get_engine(DEFAULT_POLICY_PATH)
                        masked_text = process_chat_uploaded_file(uploaded_chat_file, engine)
                        chunks = chunk_text(masked_text)
                        from langchain_core.documents import Document
                        custom_docs = [
                            Document(page_content=chk, metadata={"document_category": "custom", "source": uploaded_chat_file.name})
                            for chk in chunks
                        ]
                    
                    pipeline = get_secure_rag_pipeline()
                    res = pipeline.execute_query(user_id, query, custom_docs=custom_docs, session_id=active_sid)
                
                response_text = res["response"]
                st.write(response_text)
                
                # Save RAG metrics to state
                new_idx = len(st.session_state.chat_history)
                st.session_state.rag_history[new_idx] = {
                    "original_query": query,
                    "masked_query": res["masked_query"],
                    "sources": res["sources"],
                    "blocked": res["blocked"],
                    "event_id": res["event_id"]
                }
                        
            else:
                # Chatbot Mode
                bot = get_general_chatbot()
                response_text = bot.process_message(user_id, query, session_id=active_sid)
                st.write(response_text)

    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
    st.rerun()
