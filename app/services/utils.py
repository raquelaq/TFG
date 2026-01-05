import re
import json
import os
#from google.generativeai import genai
from ..config import *
from ..services.gemini import *

from fastapi import Request, Header, HTTPException, Depends

import jwt
import requests
from cryptography.x509 import load_pem_x509_certificate
from cryptography.hazmat.backends import default_backend

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

KB_PATH = os.path.join(DATA_DIR, "KnowledgeBase.json")
CONVERSATION_STORE_PATH = os.path.join(DATA_DIR, "conversation_store.json")

def convert_markdown_for_google_chat(text: str) -> str:
    text = re.sub(r'\s+\*\s+', '\n- ', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)\*(?!\*)', r'_\1_', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\2', text)
    return text

def get_conversation(chat_id: str):
    if not os.path.exists(CONVERSATION_STORE_PATH):
        return []

    try:
        with open(CONVERSATION_STORE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get(chat_id, [])
    except json.JSONDecodeError:
        print("❌ conversation_store.json corrupto")
        return []


def save_conversation(chat_id: str, conversation):
    os.makedirs(os.path.dirname(CONVERSATION_STORE_PATH), exist_ok=True)

    try:
        with open(CONVERSATION_STORE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                data = {}
    except Exception:
        data = {}

    data[chat_id] = conversation

    tmp_path = CONVERSATION_STORE_PATH + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    os.replace(tmp_path, CONVERSATION_STORE_PATH)

def delete_converation_cache():
    if os.path.exists(KB_PATH):
        with open(KB_PATH + 'conversation_store.json', 'w') as f:
            json.dump({}, f, indent=4, ensure_ascii=False)

def delete_conversation_cache_user(user=None):
    if user:
        conversation_file_path = os.path.join(KB_PATH, 'conversation_store.json')
        if os.path.exists(conversation_file_path):
            try:
                with open(conversation_file_path, 'r') as f:
                    conversation_data = json.load(f)
            except json.JSONDecodeError:
                conversation_data = {}
            if user in conversation_data:
                del conversation_data[user]
                with open(conversation_file_path, 'w') as f:
                    json.dump(conversation_data, f, indent=4, ensure_ascii=False)  # Using indent for readability
                print(f"User '{user}' conversation data removed successfully.")
                return True
            else:
                print(f"User '{user}' not found in conversation cache.")
                return False
        else:
            print("Conversation store file does not exist.")
            return False
    else:
        print("No user specified for deletion.")
        return False

def api_key_guard(request: Request, x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")

    api_data = API_KEYS.get(x_api_key)
    if not api_data:
        raise HTTPException(status_code=403, detail="Invalid API key")

    path = request.url.path

    if path not in api_data["allowed_endpoints"]:
        raise HTTPException(status_code=403, detail="API key not allowed for this endpoint")

    return api_data

def get_google_chat_certificates():
    url = "https://www.googleapis.com/service_accounts/v1/metadata/x509/chat@system.gserviceaccount.com"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def cert_to_public_key(cert_str):
    cert_obj = load_pem_x509_certificate(cert_str.encode(), default_backend())
    return cert_obj.public_key()

def verify_google_chat_token(token: str, expected_audience: list[str]):
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header["kid"]

        unverified_payload = jwt.decode(token, options={"verify_signature": False})
        aud = unverified_payload.get("aud")

        if aud not in expected_audience:
            print(f"❌ Audience '{aud}' not in expected list")
            return None

        certs = get_google_chat_certificates()
        if kid not in certs:
            print(f"❌ Certificate with key ID {kid} not found")
            return None

        cert_str = certs[kid]
        public_key = cert_to_public_key(cert_str)

        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=aud,
            issuer="chat@system.gserviceaccount.com",
            leeway=60
        )
        return payload

    except jwt.ExpiredSignatureError:
        print("❌ Token expired")
    except jwt.InvalidTokenError as e:
        print(f"❌ Invalid token: {e}")
    except Exception as ex:
        print(f"❌ Error verifying token: {ex}")
    return None