from google.oauth2 import service_account
from google.cloud import bigquery
from dotenv import load_dotenv

import pytz
import json
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CREDS = os.getenv("CREDENTIALS")

if not CREDS:
    raise ValueError("Missing Google credentials!")

try:
    CREDS_FILE = json.loads(CREDS)
except json.JSONDecodeError as e:
    raise ValueError("Invalid JSON in the credentials env variable") from e

# Credentials and client
SCOPE = [
    'https://www.googleapis.com/auth/bigquery',
    'https://www.googleapis.com/auth/drive'
]
GOOGLE_CREDS = service_account.Credentials.from_service_account_info(
    CREDS_FILE,
    scopes=SCOPE
)
BQ_CLIENT = bigquery.Client(credentials=GOOGLE_CREDS, project=GOOGLE_CREDS.project_id)

MNL_TZ = pytz.timezone("Asia/Manila")