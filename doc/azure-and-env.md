Title: Azure API and .env (simple)

1. Get Azure keys:
- In the Azure portal create the service you need (e.g., Cognitive Services).
- Copy the `Endpoint` and the `Key` from the resource page.

2. Create a file named `.env` in the project root (do not commit it).

3. Put these lines in `.env` (replace values):

AZURE_ENDPOINT=https://your-endpoint.azure.com
AZURE_API_KEY=your_api_key_here

4. Keep `.env` private. Add `.env` to `.gitignore` if not already ignored.

5. (Optional) Install `python-dotenv` to load `.env` in Python:

pip install python-dotenv

6. In Python, read the vars like this:

from dotenv import load_dotenv
import os

load_dotenv()
endpoint = os.getenv("AZURE_ENDPOINT")
key = os.getenv("AZURE_API_KEY")

7. Test the values by printing or logging them (do not print keys in public logs).
