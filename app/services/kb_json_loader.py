import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KB_JSON_PATH = os.getenv(
    "KB_JSON_PATH",
    os.path.join(BASE_DIR, "data", "KnowledgeBase.json")
)

def load_kb_from_json(file_path: str = KB_JSON_PATH) -> list:
    try:
        with open(file_path, "r", encoding= "utf_8") as f:
            return json.load(f)
    except Exception as e:
        print(e)
        return []