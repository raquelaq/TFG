import re
import json
import os
from datetime import datetime 
from ..config import *

from fastapi import Request, Header, HTTPException, Depends

import jwt
import requests
from cryptography.x509 import load_pem_x509_certificate
from cryptography.hazmat.backends import default_backend

def convert_markdown_for_google_chat(text: str) -> str:
    text = re.sub(r'\s+\*\s+', '\n- ', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)\*(?!\*)', r'_\1_', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\2', text)
    return text

def get_conversation(chat_id: str):
    if os.path.exists(DATA_STORE):
        with open(DATA_STORE + 'conversation_store.json', 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}  # Si el archivo está vacío o mal formado, inicializamos con un diccionario vacío
            return data.get(chat_id, [])
    return []

def save_conversation(chat_id: str, conversation):
    if os.path.exists(DATA_STORE):
        with open(DATA_STORE + 'conversation_store.json', 'r') as f:
            data = json.load(f)
    else:
        data = {}
    data[chat_id] = conversation
    with open(DATA_STORE + 'conversation_store.json', 'w') as f:
        json.dump(data, f)

def delete_converation_cache():
    if os.path.exists(DATA_STORE):
        with open(DATA_STORE + 'conversation_store.json', 'w') as f:
            json.dump({}, f)

def read_kb_file() -> str:    
    kb_file_path = DATA_STORE + 'kb.txt'
    if os.path.exists(kb_file_path):
        with open(kb_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def write_kb_file(content: str):
    print(DATA_STORE + 'kb.txt')
    kb_file_path = DATA_STORE + 'kb.txt'
    with open(kb_file_path, 'w', encoding='utf-8') as f:
        f.write(content)



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







#############################
#   VALIDAMOS GOOGLE AUTH   #
#############################

def get_google_chat_certificates():
    url = "https://www.googleapis.com/service_accounts/v1/metadata/x509/chat@system.gserviceaccount.com"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def cert_to_public_key(cert_str):
    cert_obj = load_pem_x509_certificate(cert_str.encode(), default_backend())
    return cert_obj.public_key()

def verify_google_chat_token(token: str, expected_audience: str):
    try:
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header["kid"]

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
            audience=expected_audience,
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
#######################################################