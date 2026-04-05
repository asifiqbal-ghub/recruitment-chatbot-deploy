Title: Create and use a Python virtual environment (Windows)

1. Create the virtual environment in the project folder:

python -m venv .venv

2. Activate it (PowerShell):

.\.venv\Scripts\Activate.ps1

Or (Command Prompt):

.\.venv\Scripts\activate.bat

3. Install project packages:

pip install -r requirements.txt

4. When done, deactivate:

deactivate

Notes:
- Use the same commands on other machines to reproduce the environment.
- If you are on macOS/Linux, replace `\.venv\Scripts\Activate.ps1` with
  `source .venv/bin/activate`.
