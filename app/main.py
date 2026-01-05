from dotenv import load_dotenv
load_dotenv()

import os
from configparser import ConfigParser
import json

from fastapi import FastAPI
from app.routes.chat import router as chat_router
from app.services.KnowledgeBaseFiltering import initialize_model_and_kb
from app.config import DATA_DIR

#if os.path.exists('config.ini'):
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.ini')
if os.path.exists(CONFIG_PATH):
    config = ConfigParser()
    #config.read("config.ini", encoding='utf-8')
    config.read(CONFIG_PATH, encoding='utf-8')

    DATA_STORE = config.get("GENERAL", "DATA_STORE")
    GOOGLE_CLIENT_EMAIL = config.get("GENERAL", "GOOGLE_CLIENT_EMAIL")
    GOOGLE_PRIVATE_KEY = config.get("GENERAL", "GOOGLE_PRIVATE_KEY").replace('\\n', '\n')
    AUDIENCE = config.get("GENERAL", "AUDIENCE")
    ID_DRIVE_KB = config.get("GENERAL", "ID_DRIVE_KB")
    GEMINI_API_KEY = config.get("GENERAL", "GEMINI_API_KEY")
    JIRA_AUTH_HEADER = config.get("GENERAL", "JIRA_AUTH_HEADER")

else:
    DATA_STORE = os.getenv("DATA_STORE")
    GOOGLE_CLIENT_EMAIL = os.getenv("GOOGLE_CLIENT_EMAIL")
    GOOGLE_PRIVATE_KEY = os.getenv("GOOGLE_PRIVATE_KEY").replace('\\n', '\n')
    AUDIENCE = os.getenv("AUDIENCE")
    ID_DRIVE_KB = os.getenv("GOOGLE_CLIENT_EMAIL")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    JIRA_AUTH_HEADER = os.getenv("JIRA_AUTH_HEADER")


if os.path.exists("api_keys.json"):
    try:
        with open("api_keys.json", "r", encoding="utf-8") as f:
            API_KEYS = json.load(f)
    except json.JSONDecodeError:
        print("❌ api_keys.json corrupto")
        API_KEYS = {}
else:
    API_KEYS = json.loads(os.getenv("API_KEYS", "{}"))


initialize_model_and_kb(str(DATA_DIR / "kb_embeddings.json"))

# Crear instancia FastAPI
app = FastAPI(
    title="API Soporte Técnico",
    version="1.0"
)

# Registrar las rutas del chatbot
app.include_router(chat_router)

@app.get("/")
def root():
    return {"message": "✔️ API del asistente corporativo funcionando correctamente"}