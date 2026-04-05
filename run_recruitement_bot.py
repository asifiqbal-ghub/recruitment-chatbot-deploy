import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# --------------------------------------------------
# Load Excel Data
# --------------------------------------------------
DATA_DIR = r"C:\Users\aisiq\OneDrive\Desktop\Recruitment_Data_Analysis" # Change this based on your file location

requirement_df = pd.read_excel(os.path.join(DATA_DIR, "Requirement_Table_100.xlsx"))
interview_df   = pd.read_excel(os.path.join(DATA_DIR, "Interview_Table_100.xlsx"))
recruiter_df   = pd.read_excel(os.path.join(DATA_DIR, "Recruiter_Table_100.xlsx"))
offer_df       = pd.read_excel(os.path.join(DATA_DIR, "Offer_Table_100.xlsx"))
candidate_df   = pd.read_excel(os.path.join(DATA_DIR, "Candidate_Table_100.xlsx"))
application_df = pd.read_excel(os.path.join(DATA_DIR, "Application_Table_100.xlsx"))

# --------------------------------------------------
# Data Preparation
# --------------------------------------------------
recruiter_df = recruiter_df.rename(columns={"status": "recruiter_status"})

df_merged = (
    application_df
    .merge(candidate_df, on="candidate_id", how="left")
    .merge(requirement_df, on="requirement_id", how="left", suffixes=("", "_requirement"))
    .merge(interview_df, on="application_id", how="left")
    .merge(offer_df, left_on="candidate_id", right_on="offer_candidate_id", how="left")
    .merge(
        recruiter_df,
        left_on="screened_by_recruiter_id",
        right_on="recruiter_id",
        how="left",
        suffixes=("", "_recruiter")
    )
)

df_merged.rename(columns={
    "full_name": "Candidate_full_name",
    "email_x": "Candidate_email",
    "name": "Recruiter_name",
    "email_y": "Recruiter_email"
}, inplace=True)

df_json = df_merged.to_json(orient="records")

# --------------------------------------------------
# Azure OpenAI Call
# --------------------------------------------------
headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_API_KEY
}

def ask_azure_openai(question: str) -> str:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert recruitment data analyst. Use the provided dataset to answer accurately."
            },
            {
                "role": "system",
                "content": f"Recruitment dataset (JSON):\n{df_json}"
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2000
    }

    response = requests.post(
        AZURE_OPENAI_ENDPOINT,
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()

    return f"Error {response.status_code}: {response.text}"

# --------------------------------------------------
# CLI Chat Loop
# --------------------------------------------------
if __name__ == "__main__":
    print("Recruitment Chatbot is Ready!")
    while True:
        user_question = input("\nAsk your recruiting question: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        answer = ask_azure_openai(user_question)
        print("\nAnswer:", answer)