
import re
import json
import os
from datetime import datetime 
from ..config import DATA_STORE

def convert_markdown_for_google_chat(text: str) -> str:
    text = re.sub(r'\s+\*\s+', '\n- ', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.*?)\*(?!\*)', r'_\1_', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\2', text)
    return text

def get_conversation(chat_id: str):
    if os.path.exists(DATA_STORE):
        with open(DATA_STORE, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}  # Si el archivo está vacío o mal formado, inicializamos con un diccionario vacío
            return data.get(chat_id, [])
    return []

def save_conversation(chat_id: str, conversation):
    if os.path.exists(DATA_STORE):
        with open(DATA_STORE, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    data[chat_id] = conversation
    with open(DATA_STORE, 'w') as f:
        json.dump(data, f)

def delete_converation_cache():
    if os.path.exists(DATA_STORE):
        with open(DATA_STORE, 'w') as f:
            json.dump({}, f)
