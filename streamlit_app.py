import io
import os
import tempfile
from pathlib import Path
import streamlit as st

from app.data_masking.masking_engine import MaskingEngine
from app.data_masking.masking_policy import MaskingPolicy
from app.data_masking.file_processors import FileProcessor
from app.rag.secure_rag import SecureRAGPipeline
from app.chatbot.chatbot import SecuredChatbot
from app.auth.rbac import RBAC

DEFAULT_POLICY_PATH = "data/masking_policies/mg_policy.yaml"

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

st.set_page_config(page_title="Secure Chat & Masking Hub", layout="wide", page_icon="🛡️")

# Custom CSS to center title/caption, increase font size globally, and use standard neutral colors
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.8rem !important;
        font-weight: 800;
        margin-top: 1rem;
        margin-bottom: 0.1rem;
    }
    .main-caption {
        text-align: center;
        font-size: 1.25rem !important;
        color: #888888;
        margin-bottom: 2rem;
    }
    /* Scale up the font size of the entire page */
    html, body, p, div, span, label, select, button, input, textarea, .stText, .stMarkdown, [class*="css"] {
        font-size: 1.15rem !important;
    }
    h1, h2, h3 {
        font-size: 1.65rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">ChatBot and Data Masking</div>', unsafe_allow_html=True)

# Simple center-aligned tab setup
tab_chat, tab_masker = st.tabs([" Chatbot", " File Masker"])

# =========================================================================
# TAB 1: INTERACTIVE SECURE CHATBOT / RAG
# =========================================================================
with tab_chat:
    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.subheader("Settings")
        
        # User selector mapping standard accounts
        user_option = st.selectbox(
            "Select User Identity",
            options=[
                "1 - Admin",
                "2 - HR User",
                "3 - Finance User",
                "4 - IT User",
                "5 - Guest"
            ]
        )
        
        # Extract operational account mappings
        user_mapping = {
            "1 - Admin": "1",
            "2 - HR User": "2",
            "3 - Finance User": "3",
            "4 - IT User": "4",
            "5 - Guest": "5"
        }
        user_id = user_mapping[user_option]
        user_role = RBAC.get_role(user_id) or "guest"

        st.info(f"**Current Role**: `{user_role.upper()}`")

        # Mode Selection
        chat_mode = st.radio(
            "Select Chat Mode",
            ["Secure RAG", "General Chatbot"]
        )

        # RAG Context Source Selection
        context_source = "Standard Knowledge Base"
        uploaded_chat_file = None
        if chat_mode == "Secure RAG":
            st.markdown("---")
            context_source = st.radio(
                "RAG Context Source",
                ["Standard Knowledge Base", "Uploaded Document"]
            )
            if context_source == "Uploaded Document":
                uploaded_chat_file = st.file_uploader(
                    "Upload Chat Context File",
                    type=["pdf", "csv", "xlsx", "xls", "docx", "txt", "json"],
                    key="chat_file_uploader"
                )
                if uploaded_chat_file is not None:
                    st.success(f"Loaded: `{uploaded_chat_file.name}`")

        clear_btn = st.button("Clear Conversation History", type="secondary")
        if clear_btn:
            st.session_state.chat_history = []
            st.session_state.rag_history = {}
            st.rerun()

    with col_right:
        st.subheader("Chat Terminal")
        
        # Initialize conversation session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "rag_history" not in st.session_state:
            st.session_state.rag_history = {}

        # Render conversation history
        for idx, chat in enumerate(st.session_state.chat_history):
            with st.chat_message(chat["role"]):
                st.write(chat["content"])
                
                # Render metadata/security logs for assistant messages
                if chat["role"] == "assistant" and idx in st.session_state.rag_history:
                    rag_data = st.session_state.rag_history[idx]
                    
                    # Simple inspection
                    if user_role == "admin":
                        with st.expander(" View Security Details"):
                            st.markdown(f"**Original Query**: `{rag_data['original_query']}`")
                            st.markdown(f"**Masked Query**: `{rag_data['masked_query']}`")
                            st.markdown(f"**Blocked**: `{rag_data['blocked']}`")
                            st.markdown(f"**Event ID**: `{rag_data['event_id']}`")

        # Chat Input
        query = st.chat_input("Ask a question about policies, expenses, or MG Motors...")
        
        if query:
            # 1. Display User Message
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            # 2. Process Bot Response
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
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
                            res = pipeline.execute_query(user_id, query, custom_docs=custom_docs)
                        
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
                        response_text = bot.process_message(user_id, query)
                        st.write(response_text)

            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            st.rerun()

# =========================================================================
# TAB 2: DOCUMENT DATA MASKING TOOL
# =========================================================================
with tab_masker:
    st.subheader("Anonymizer Configuration")
    
    policy_path = st.text_input("Active Masking Policy File Path", value=DEFAULT_POLICY_PATH)
    uploaded = st.file_uploader("Select File to Anonymize", type=["pdf", "csv", "xlsx", "xls", "docx", "txt", "json"], accept_multiple_files=False)
    run_btn = st.button("Anonymize Uploaded File", type="primary", disabled=uploaded is None)

    if run_btn and uploaded is not None:
        if not Path(policy_path).exists():
            st.error(f"Policy file not found: {policy_path}")
            st.stop()

        engine = get_engine(policy_path)

        with st.spinner("Anonymizing content..."):
            suffix = Path(uploaded.name).suffix.lower()
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / f"upload{suffix}"
                tmp_path.write_bytes(uploaded.getvalue())

                if suffix == ".txt":
                    masked_text = FileProcessor.process_txt(tmp_path, engine)
                elif suffix == ".pdf":
                    masked_text = FileProcessor.process_pdf(tmp_path, engine)
                elif suffix == ".docx":
                    masked_text = FileProcessor.process_docx(tmp_path, engine)
                elif suffix == ".csv":
                    masked_text = FileProcessor.process_csv(tmp_path, engine)
                elif suffix in [".xls", ".xlsx"]:
                    masked_text = FileProcessor.process_excel(tmp_path, engine)
                elif suffix == ".json":
                    masked_text = FileProcessor.process_json(tmp_path, engine)
                else:
                    masked_text = engine.mask_text(uploaded.getvalue().decode("utf-8", errors="ignore")).masked_text

        st.subheader("Masked Document Output")
        st.text_area("Anonymized content text block", value=masked_text, height=420)

        out_bytes = masked_text.encode("utf-8")
        out_name = f"{Path(uploaded.name).stem}_masked{suffix if suffix != '.xlsx' and suffix != '.xls' else '.csv'}"

        st.download_button(
            label="Download Masked File Content",
            data=out_bytes,
            file_name=out_name,
            mime="text/plain",
        )
