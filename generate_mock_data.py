"""
Run this script to generate mock recruitment Excel files.
Output: 6 .xlsx files in a folder called mock_data/
"""

import os
import random
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
OUTPUT_DIR = "mock_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def rand_date(start_days_ago=180, end_days_ago=0):
    start = datetime.today() - timedelta(days=start_days_ago)
    end   = datetime.today() - timedelta(days=end_days_ago)
    return (start + timedelta(days=random.randint(0, (end - start).days))).strftime("%Y-%m-%d")

N_RECRUITERS   = 10
N_REQUIREMENTS = 20
N_CANDIDATES   = 100
N_APPLICATIONS = 100
N_INTERVIEWS   = 80
N_OFFERS       = 40

# --------------------------------------------------
# 1. Recruiter Table
# --------------------------------------------------
recruiter_names  = [
    "Alice Johnson", "Bob Smith", "Carla Mendes", "David Lee", "Eva Brown",
    "Faisal Malik", "Grace Kim", "Henry White", "Iris Patel", "Jake Turner"
]
recruiter_emails = [n.lower().replace(" ", ".") + "@company.com" for n in recruiter_names]

recruiter_df = pd.DataFrame({
    "recruiter_id":   [f"R{str(i).zfill(3)}" for i in range(1, N_RECRUITERS + 1)],
    "name":           recruiter_names,
    "email":          recruiter_emails,
    "phone":          [f"+1-555-{random.randint(1000,9999)}" for _ in range(N_RECRUITERS)],
    "department":     random.choices(["Engineering", "Sales", "HR", "Finance", "Marketing"], k=N_RECRUITERS),
    "status":         random.choices(["active", "active", "active", "inactive"], k=N_RECRUITERS),
})

# --------------------------------------------------
# 2. Requirement Table (Job Openings)
# --------------------------------------------------
job_titles = [
    "Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer",
    "Frontend Developer", "Backend Developer", "QA Engineer", "HR Manager",
    "Sales Executive", "Business Analyst", "ML Engineer", "Cloud Architect",
    "UX Designer", "Marketing Specialist", "Finance Analyst", "Scrum Master",
    "Technical Writer", "Security Engineer", "Database Administrator", "Support Engineer"
]
requirement_df = pd.DataFrame({
    "requirement_id":       [f"REQ{str(i).zfill(3)}" for i in range(1, N_REQUIREMENTS + 1)],
    "job_title":            job_titles,
    "department":           random.choices(["Engineering", "Sales", "HR", "Finance", "Marketing"], k=N_REQUIREMENTS),
    "location":             random.choices(["New York", "San Francisco", "Austin", "Remote", "Chicago"], k=N_REQUIREMENTS),
    "required_skills":      random.choices(["Python, SQL", "React, JS", "Java, Spring", "AWS, Docker", "Excel, PowerBI"], k=N_REQUIREMENTS),
    "experience_required":  random.choices(["0-2 years", "2-4 years", "4-7 years", "7+ years"], k=N_REQUIREMENTS),
    "openings":             random.choices([1, 2, 3], k=N_REQUIREMENTS),
    "status":               random.choices(["open", "open", "open", "closed"], k=N_REQUIREMENTS),
    "posted_date":          [rand_date(180, 30) for _ in range(N_REQUIREMENTS)],
})

# --------------------------------------------------
# 3. Candidate Table
# --------------------------------------------------
first_names = ["Aisha","Rohan","Neha","James","Sara","Ali","Emily","Chen","Priya","Marcus",
               "Lily","Omar","Sofia","David","Zara","Kevin","Mia","Raj","Nina","Lucas",
               "Hannah","Yusuf","Chloe","Arjun","Isabel","Sam","Leila","Noah","Fatima","Ethan",
               "Mei","Jordan","Anika","Diego","Layla","Tyler","Riya","Chris","Nadia","Ben",
               "Tara","Mikhail","Zoe","Kiran","Elena","Abdul","Maya","Finn","Shreya","Oscar",
               "Amara","Leo","Hana","Ravi","Claire","Mohammed","Jade","Patrick","Sunita","Jack",
               "Amelia","Vikram","Ruby","Carlos","Pooja","Ryan","Nora","Siddharth","Ella","Felix",
               "Divya","Aaron","Iris","Tariq","Luna","Brendan","Nisha","Cole","Alina","Max",
               "Parveen","Sienna","Rahul","Stella","Ivan","Kavya","Hugo","Dana","Shawn","Meera",
               "Troy","Lena","Aditi","Miles","Yasmin","Caleb","Trisha","Elias","Rekha","Seth"]
last_names  = ["Gupta","Sharma","Kapoor","Smith","Williams","Khan","Johnson","Wu","Patel","Davis",
               "Chen","Hassan","Torres","Lee","Ahmed","Brown","Singh","Martinez","Kim","Wilson",
               "Nguyen","Ali","Garcia","Mehta","Lopez","Taylor","Malik","Anderson","Reddy","Thomas",
               "Liu","Harris","Roy","Hernandez","Begum","Robinson","Shah","Clark","Ivanova","Walker",
               "Joshi","Petrov","Evans","Nair","Moreau","Siddiqui","Scott","Murphy","Kaur","White",
               "Okafor","Green","Tanaka","Verma","Baker","El-Amin","Adams","O'Brien","Pillai","Carter",
               "Turner","Banerjee","Hill","Rivera","Desai","Mitchell","Andersen","Chatterjee","Campbell","Fischer",
               "Iyer","Nelson","Johansson","Farooq","Stewart","Collins","Bose","Morris","Volkov","Rogers",
               "Krishnan","Bell","Das","Ward","Sokolov","Menon","Cooper","Hussain","Reed","Nair",
               "Fleming","Weber","Ghosh","Ramirez","Abubakar","Nielsen","Soni","Hoffman","Saxena","Perry"]

full_names = [f"{first_names[i]} {last_names[i]}" for i in range(N_CANDIDATES)]
emails     = [f"{first_names[i].lower()}.{last_names[i].lower()}{i}@email.com" for i in range(N_CANDIDATES)]

candidate_df = pd.DataFrame({
    "candidate_id":        [f"C{str(i).zfill(3)}" for i in range(1, N_CANDIDATES + 1)],
    "full_name":           full_names,
    "email":               emails,
    "phone":               [f"+1-555-{random.randint(1000,9999)}" for _ in range(N_CANDIDATES)],
    "location":            random.choices(["New York", "San Francisco", "Austin", "Remote", "Chicago", "Boston"], k=N_CANDIDATES),
    "years_of_experience": [random.randint(0, 15) for _ in range(N_CANDIDATES)],
    "skills":              random.choices(["Python, SQL", "React, JS", "Java, Spring", "AWS, Docker", "Excel, PowerBI", "C++, Linux"], k=N_CANDIDATES),
    "education":           random.choices(["Bachelor's", "Master's", "PhD", "Diploma"], k=N_CANDIDATES),
    "source":              random.choices(["LinkedIn", "Indeed", "Referral", "Company Website", "Recruiter"], k=N_CANDIDATES),
})

# --------------------------------------------------
# 4. Application Table
# --------------------------------------------------
candidate_ids   = candidate_df["candidate_id"].tolist()
requirement_ids = requirement_df["requirement_id"].tolist()
recruiter_ids   = recruiter_df["recruiter_id"].tolist()

app_statuses = ["Applied", "Screening", "Interview Scheduled", "Offer Extended", "Hired", "Rejected"]

application_df = pd.DataFrame({
    "application_id":           [f"APP{str(i).zfill(3)}" for i in range(1, N_APPLICATIONS + 1)],
    "candidate_id":             random.sample(candidate_ids, N_APPLICATIONS),
    "requirement_id":           random.choices(requirement_ids, k=N_APPLICATIONS),
    "screened_by_recruiter_id": random.choices(recruiter_ids, k=N_APPLICATIONS),
    "application_date":         [rand_date(150, 0) for _ in range(N_APPLICATIONS)],
    "status":                   random.choices(app_statuses, weights=[15, 20, 25, 15, 15, 10], k=N_APPLICATIONS),
    "resume_score":             [round(random.uniform(50, 100), 1) for _ in range(N_APPLICATIONS)],
    "notes":                    random.choices(["Strong profile", "Needs review", "Good cultural fit", "Overqualified", "Follow up needed", ""], k=N_APPLICATIONS),
})

# --------------------------------------------------
# 5. Interview Table (subset of applications)
# --------------------------------------------------
interview_app_ids = random.sample(application_df["application_id"].tolist(), N_INTERVIEWS)

interview_df = pd.DataFrame({
    "interview_id":    [f"INT{str(i).zfill(3)}" for i in range(1, N_INTERVIEWS + 1)],
    "application_id":  interview_app_ids,
    "interview_date":  [rand_date(120, 0) for _ in range(N_INTERVIEWS)],
    "interview_type":  random.choices(["Phone Screen", "Technical", "HR Round", "Final Round"], k=N_INTERVIEWS),
    "interviewer":     random.choices(recruiter_names, k=N_INTERVIEWS),
    "outcome":         random.choices(["Pass", "Fail", "Pending"], weights=[50, 25, 25], k=N_INTERVIEWS),
    "feedback":        random.choices(["Excellent communication", "Strong technical skills", "Needs improvement", "Good problem solver", "Culture fit confirmed", ""], k=N_INTERVIEWS),
    "rating":          [random.randint(1, 5) for _ in range(N_INTERVIEWS)],
})

# --------------------------------------------------
# 6. Offer Table (subset of candidates)
# --------------------------------------------------
offer_candidate_ids = random.sample(candidate_ids, N_OFFERS)

offer_df = pd.DataFrame({
    "offer_id":           [f"OFF{str(i).zfill(3)}" for i in range(1, N_OFFERS + 1)],
    "offer_candidate_id": offer_candidate_ids,
    "salary_offered":     [random.randint(60000, 180000) for _ in range(N_OFFERS)],
    "currency":           ["USD"] * N_OFFERS,
    "offer_date":         [rand_date(90, 0) for _ in range(N_OFFERS)],
    "offer_status":       random.choices(["Accepted", "Rejected", "Pending", "Negotiating"], weights=[50, 20, 20, 10], k=N_OFFERS),
    "joining_date":       [rand_date(0, -60) for _ in range(N_OFFERS)],
    "benefits":           random.choices(["Health + 401k", "Health + Stock", "Health only", "Full package"], k=N_OFFERS),
})

# --------------------------------------------------
# Write Excel files
# --------------------------------------------------
files = {
    "Requirement_Table_100.xlsx": requirement_df,
    "Candidate_Table_100.xlsx":   candidate_df,
    "Application_Table_100.xlsx": application_df,
    "Interview_Table_100.xlsx":   interview_df,
    "Offer_Table_100.xlsx":       offer_df,
    "Recruiter_Table_100.xlsx":   recruiter_df,
}

for filename, df in files.items():
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_excel(path, index=False)
    print(f"✅ {filename} — {len(df)} rows")

print(f"\nAll files saved to ./{OUTPUT_DIR}/")
