import os
import streamlit as st
import pandas as pd
from openai import AzureOpenAI

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="File Q&A Chatbot", page_icon="📁", layout="wide")
st.title("📁 File Q&A Chatbot")
st.caption("Upload any files and ask questions about their content.")

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
# Sidebar: File Upload
# --------------------------------------------------
st.sidebar.header("📂 Upload Files")
st.sidebar.markdown(f"Supported formats: `{', '.join(SUPPORTED_TYPES)}`")

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
                st.sidebar.success(f"✅ Loaded {len(uploaded_files)} file(s).")
            except Exception as e:
                st.sidebar.error(f"Error processing files: {e}")
                st.stop()
    else:
        st.sidebar.success(f"✅ {len(uploaded_files)} file(s) ready.")

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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your files..."):
    if not files_ready or "files_context" not in st.session_state:
        st.warning("Please upload at least one file using the sidebar first.")
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
