Title: Setup CSV for candidates

1. Create a folder named `data` in the project root if it does not exist.

2. Create a file named `candidates.csv` inside `data`.

3. The CSV must have a header row with these columns (exact names):
- Name
- Position
- Status

4. Example content (first line is the header):

Name,Position,Status
Aisha Gupta,Software Engineer,Screening
Rohan Sharma,Data Scientist,Interview Scheduled
Neha Kapoor,Frontend Developer,Offer Extended

5. Save with UTF-8 encoding. Do not add extra hidden columns.

That's it — the project reads `data/candidates.csv`.
