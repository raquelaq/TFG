from google.oauth2 import service_account
from googleapiclient.discovery import build
from ..config import GOOGLE_CLIENT_EMAIL, GOOGLE_PRIVATE_KEY

SCOPES = ['https://www.googleapis.com/auth/documents', 'https://www.googleapis.com/auth/drive']

def read_google_doc(doc_id: str) -> str:
    try:
        credentials = service_account.Credentials.from_service_account_info({
            "private_key": GOOGLE_PRIVATE_KEY,
            "client_email": GOOGLE_CLIENT_EMAIL,
            "token_uri": "https://oauth2.googleapis.com/token",
            "type": "service_account"
        }, scopes=SCOPES)

        service = build('docs', 'v1', credentials=credentials)
        document = service.documents().get(documentId=doc_id).execute()
        return "\n".join([
            el.get("textRun", {}).get("content", "")
            for c in document.get("body", {}).get("content", [])
            for el in c.get("paragraph", {}).get("elements", [])
        ])
    except Exception as e:
        print(f"Error reading Google Doc: {e}")
        raise e
