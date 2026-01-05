import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "..", "data", "users.json")
USERS_FILE = os.path.normpath(USERS_FILE)

def load_users():
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("users", [])
    except json.JSONDecodeError:
        print("❌ users.json corrupto")
        return []
    except Exception as e:
        print(f"❌ Error cargando usuarios: {e}")
        return []


def authenticate(email: str, password: str):
    users = load_users()
    for u in users:
        if u["email"] == email and u["password"] == password:
            return u
    return None


def get_role(email: str):
    users = load_users()
    for u in users:
        if u["email"] == email:
            return u["role"]
    return None