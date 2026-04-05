import os
import streamlit as st
import pandas as pd
from openai import AzureOpenAI

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Recruitment Chatbot", page_icon="💼", layout="wide")

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a8b2c1;
        font-size: 1rem;
        margin: 0.4rem 0 0 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #f8f9fc;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid #e53e3e;
        color: #e53e3e;
        background: transparent;
        font-weight: 500;
        transition: all 0.2s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #e53e3e;
        color: white;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 10px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.5rem;
        padding: 0.25rem;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #718096;
    }
    .empty-state .icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }
    .empty-state h3 {
        font-size: 1.3rem;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 0.5rem;
    }

    /* Stat badges in sidebar */
    .stat-badge {
        background: #ebf8ff;
        border: 1px solid #bee3f8;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        font-size: 0.85rem;
        color: #2b6cb0;
        margin-top: 0.5rem;
    }

    /* Divider */
    hr { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# API Credentials
# --------------------------------------------------
AZURE_OPENAI_API_KEY     = st.secrets.get("AZURE_OPENAI_API_KEY", "")     or os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT    = st.secrets.get("AZURE_OPENAI_ENDPOINT", "")    or os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_MODEL       = st.secrets.get("AZURE_OPENAI_MODEL", "")       or os.getenv("AZURE_OPENAI_MODEL", "")
AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "") or os.getenv("AZURE_OPENAI_API_VERSION", "")

if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_MODEL, AZURE_OPENAI_API_VERSION]):
    st.error(
        "Azure OpenAI credentials not found. Add AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
        "AZURE_OPENAI_MODEL, and AZURE_OPENAI_API_VERSION in Streamlit Cloud Secrets or environment variables."
    )
    st.stop()

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

# --------------------------------------------------
# File Content Extraction
# --------------------------------------------------
SUPPORTED_TYPES = ["xlsx", "xls", "csv", "txt", "md", "pdf", "docx"]


def extract_file_content(file) -> str:
    ext = file.name.rsplit(".", 1)[-1].lower()

    if ext in ("xlsx", "xls"):
        df = pd.read_excel(file)
        return df.to_string(index=False)

    elif ext == "csv":
        df = pd.read_csv(file)
        return df.to_string(index=False)

    elif ext in ("txt", "md"):
        return file.read().decode("utf-8", errors="replace")

    elif ext == "pdf":
        import pdfplumber
        pages = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)

    elif ext == "docx":
        from docx import Document
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    else:
        return f"[Unsupported file type: {ext}]"


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>💼 Recruitment Chatbot</h1>
    <p>Upload your recruitment files and ask questions about your hiring data.</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Sidebar: File Upload
# --------------------------------------------------
st.sidebar.markdown("## 📂 Upload Recruitment Data")
st.sidebar.markdown(
    f"<small style='color:#718096'>Supported: {', '.join(f'<code>{t}</code>' for t in SUPPORTED_TYPES)}</small>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more files",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

files_ready = len(uploaded_files) > 0

# --------------------------------------------------
# Data Processing
# --------------------------------------------------
if files_ready:
    file_key = tuple(f.name for f in uploaded_files)

    if st.session_state.get("file_key") != file_key:
        with st.spinner("Processing uploaded files..."):
            try:
                context_parts = []
                for f in uploaded_files:
                    content = extract_file_content(f)
                    context_parts.append(f"### File: {f.name}\n{content}")

                st.session_state.files_context = "\n\n---\n\n".join(context_parts)
                st.session_state.file_key = file_key
                st.session_state.messages = []
            except Exception as e:
                st.sidebar.error(f"Error processing files: {e}")
                st.stop()

    st.sidebar.markdown(
        f"<div class='stat-badge'>✅ {len(uploaded_files)} file(s) loaded</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
else:
    st.sidebar.info("Upload one or more files to get started.")

# --------------------------------------------------
# Chat Interface
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🤖</div>
        <h3>Ready to help with your recruitment data</h3>
        <p>Upload your files in the sidebar, then ask anything about your hiring pipeline.</p>
    </div>
    """, unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a recruitment question..."):
    if not files_ready or "files_context" not in st.session_state:
        st.warning("Please upload at least one recruitment file using the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                system_context = (
                    "You are a helpful assistant. Answer the user's question based only on the "
                    "file content provided below. Be concise and accurate.\n\n"
                    f"File content:\n{st.session_state.files_context}"
                )

                try:
                    response = client.chat.completions.create(
                        model=AZURE_OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": system_context},
                            {"role": "user",   "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=2000,
                    )
                    answer = response.choices[0].message.content
                except Exception as e:
                    answer = f"⚠️ Error calling Azure OpenAI API: {e}"

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
