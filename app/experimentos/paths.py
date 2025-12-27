import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(APP_DIR, "data")

KB_JSON_PATH = os.path.join(DATA_DIR, "KnowledgeBase.json")
KB_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "kb_embeddings.json")
CONVERSATION_STORE_PATH = os.path.join(DATA_DIR, "conversation_store.json")
USERS_PATH = os.path.join(DATA_DIR, "users.json")
