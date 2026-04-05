import os
import streamlit as st
import pandas as pd
from openai import AzureOpenAI

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Recruitment Chatbot", page_icon="💼", layout="wide")
st.title("💼 Recruitment Chatbot")
st.caption("Upload your recruitment Excel files and ask questions about your hiring data.")

# --------------------------------------------------
# API Key
# --------------------------------------------------
AZURE_OPENAI_API_KEY     = st.secrets.get("AZURE_OPENAI_API_KEY", "")     or os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT    = st.secrets.get("AZURE_OPENAI_ENDPOINT", "")    or os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_MODEL       = st.secrets.get("AZURE_OPENAI_MODEL", "")       or os.getenv("AZURE_OPENAI_MODEL", "")
AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "") or os.getenv("AZURE_OPENAI_API_VERSION", "")

if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_MODEL, AZURE_OPENAI_API_VERSION]):
    st.error("Azure OpenAI credentials not found. Add AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_MODEL, and AZURE_OPENAI_API_VERSION in Streamlit Cloud Secrets or environment variables.")
    st.stop()

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

# --------------------------------------------------
# Sidebar: File Upload
# --------------------------------------------------
st.sidebar.header("📂 Upload Recruitment Data")
st.sidebar.markdown("Upload all 6 Excel files to enable the chatbot.")

FILE_LABELS = {
    "requirement": "Requirement Table",
    "candidate":   "Candidate Table",
    "application": "Application Table",
    "interview":   "Interview Table",
    "offer":       "Offer Table",
    "recruiter":   "Recruiter Table",
}

uploaded_files = {
    key: st.sidebar.file_uploader(f"{label} (.xlsx)", type=["xlsx"], key=key)
    for key, label in FILE_LABELS.items()
}

all_uploaded = all(f is not None for f in uploaded_files.values())

# --------------------------------------------------
# Data Processing
# --------------------------------------------------
def build_merged_df(files: dict) -> pd.DataFrame:
    requirement_df  = pd.read_excel(files["requirement"])
    candidate_df    = pd.read_excel(files["candidate"])
    application_df  = pd.read_excel(files["application"])
    interview_df    = pd.read_excel(files["interview"])
    offer_df        = pd.read_excel(files["offer"])
    recruiter_df    = pd.read_excel(files["recruiter"])

    recruiter_df = recruiter_df.rename(columns={"status": "recruiter_status"})

    df_merged = (
        application_df
        .merge(candidate_df,    on="candidate_id",    how="left")
        .merge(requirement_df,  on="requirement_id",  how="left", suffixes=("", "_requirement"))
        .merge(interview_df,    on="application_id",  how="left")
        .merge(offer_df,        left_on="candidate_id", right_on="offer_candidate_id", how="left")
        .merge(
            recruiter_df,
            left_on="screened_by_recruiter_id",
            right_on="recruiter_id",
            how="left",
            suffixes=("", "_recruiter"),
        )
    )

    df_merged.rename(columns={
        "full_name": "Candidate_full_name",
        "email_x":   "Candidate_email",
        "name":      "Recruiter_name",
        "email_y":   "Recruiter_email",
    }, inplace=True)

    return df_merged


if all_uploaded:
    # Use file names as a cache key — rebuild if files change
    file_key = tuple(f.name for f in uploaded_files.values())

    if st.session_state.get("file_key") != file_key:
        with st.spinner("Processing uploaded files..."):
            try:
                df = build_merged_df(uploaded_files)
                st.session_state.df_json  = df.to_json(orient="records")
                st.session_state.file_key = file_key
                st.session_state.messages = []  # reset chat when new data loaded
                st.sidebar.success(f"✅ Loaded {len(df):,} records.")
            except Exception as e:
                st.sidebar.error(f"Error processing files: {e}")
                st.stop()
    else:
        st.sidebar.success("✅ Data ready.")

    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
else:
    missing = [FILE_LABELS[k] for k, v in uploaded_files.items() if v is None]
    st.sidebar.warning(f"Missing: {', '.join(missing)}")

# --------------------------------------------------
# Chat Interface
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask a recruitment question..."):
    if not all_uploaded or "df_json" not in st.session_state:
        st.warning("Please upload all 6 Excel files using the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                system_context = (
                    "You are an expert recruitment data analyst. "
                    "Answer the user's question using only the recruitment dataset provided. "
                    "Be concise and accurate.\n\n"
                    f"Recruitment dataset (JSON):\n{st.session_state.df_json}"
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
