import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "..", "data", "users.json")
USERS_FILE = os.path.normpath(USERS_FILE)

def load_users():
    """Carga los usuarios desde el JSON."""
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["users"]


def authenticate(email: str, password: str):
    """
    Valida email + password.
    Devuelve:
      - None → si credenciales incorrectas
      - dict con la info del usuario → si correctas
    """
    users = load_users()
    for u in users:
        if u["email"] == email and u["password"] == password:
            return u
    return None


def get_role(email: str):
    """Devuelve el rol de un usuario ya autenticado."""
    users = load_users()
    for u in users:
        if u["email"] == email:
            return u["role"]
    return None
