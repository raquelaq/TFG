import re
import json
import os
import time
from datetime import datetime
from idlelib.iomenu import encoding
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
    if os.path.exists(KB_PATH):
        with open(CONVERSATION_STORE_PATH, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}  # Si el archivo está vacío o mal formado, inicializamos con un diccionario vacío
            return data.get(chat_id, [])
    return []

def save_conversation(chat_id: str, conversation):
    if os.path.exists(KB_PATH):
        with open(CONVERSATION_STORE_PATH, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    data[chat_id] = conversation
    with open(CONVERSATION_STORE_PATH, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

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
                # Handle case where file is empty or malformed JSON
                conversation_data = {}
            if user in conversation_data:
                del conversation_data[user]
                with open(conversation_file_path, 'w') as f:
                    json.dump(conversation_data, f, indent=4, ensure_ascii=False)  # Using indent for readability
                print(f"User '{user}' conversation data removed successfully.")
                return True  # Indicate success
            else:
                print(f"User '{user}' not found in conversation cache.")
                return False  # Indicate user not found
        else:
            print("Conversation store file does not exist.")
            return False  # Indicate file not found
    else:
        print("No user specified for deletion.")
        return False  # Indicate no user specified

def read_kb_file() -> str:    
    kb_file_path = KB_PATH + 'kb.txt'
    if os.path.exists(kb_file_path):
        with open(kb_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def write_kb_file(content: str):
    print(KB_PATH + 'kb.txt')
    kb_file_path = KB_PATH + 'kb.txt'
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

        # Extraemos el payload sin verificar la firma, solo para inspeccionar
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

        # Verificamos ahora con firma y parámetros completos
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience=aud,  # ya lo filtramos arriba
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



async def create_ticket_contents(messages):
    prompt = f"""A partir de los siguientes mensajes entre un chat de soporte y un usuario
crea un título corto descriptivo del problema que el usuario está teniendo, para crear un ticket con él.
Además, crea un resumen en el que se explique en detalle el problema del usuario para facilitar al trabajador de soporte
entender el problema del usuario.
Asegúrate de responder en formato JSON, usando los campos "title" y "summary".

Ejemplo:
{{
  "title": "Problemas conectando con Sigrid",
  "summary": "El usuario está encontrando problemas a la hora de conectarse con Sigrid. El error que está dando es el 408. Intentó acceder ignorando el firewall y el problema persiste."
}}

Base tus respuestas únicamente en el último problema de la siguiente conversación:
\"\"\"{json.dumps(messages)}\"\"\"
"""

    tools = [
        {
            "function_declarations": [
                {
                    "name": "return_json_response",
                    "description": "Devuelve una respuesta como JSON con los campos 'title' y 'summary'",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Título breve basado en la conversación."
                            },
                            "summary": {
                                "type": "string",
                                "description": "Resumen detallado del problema del usuario."
                            }
                        },
                        "required": ["title", "summary"]
                    }
                }
            ]
        }
    ]

    parsed_reply = await call_gemini_prompt(prompt)
    return parsed_reply

def respuesta_con_agente(agent, pregunta: str) -> str: # --> ESTO ES MÍO
    try:
        return agent.run(pregunta)
    except Exception as e:
        return f"⚠️ Error procesando con el agente: {e}"

#################################
# PRUEBA FILTRADO LLM
#################################


# def find_similar_incidents_gemini(user_email, user_message):
#     client = genai.Client(api_key="AIzaSyCp_J8rN857FNNRbGGCnepM_9Fly5249Jc")
#
#     try:
#         with open(KB_PATH + 'KnowledgeBase.json', 'r', encoding='latin-1') as f:
#             incidents_data = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: The file was not found.")
#         return []
#     except json.JSONDecodeError:
#         print(f"Error: Could not decode JSON from file. Check file format.")
#         return []
#
#     conversation_total = get_conversation(user_email)
#     if isinstance(conversation_total, list):
#         # Modo antiguo: solo hay conversación
#         conversation = conversation_total
#         incident_ids = []
#     elif isinstance(conversation_total, dict):
#         # Modo nuevo: hay conversación + incidencias
#         conversation = conversation_total.get("conversation", [])
#         incident_ids = conversation_total.get("Incidents", [])
#     else:
#         # Fallback por seguridad
#         conversation = []
#         incident_ids = []
#
#     conversation.append({"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()})
#
#     # Prepare incident data for the LLM
#     # We only send id, title, description_problem, and keywords_tags
#     incidents_for_llm = []
#     for incident in incidents_data:
#         incidents_for_llm.append({
#             "id": incident.get("id"),
#             "title": incident.get("title"),
#             "description_problem": incident.get("description_problem"),
#             "keywords_tags": incident.get("keywords_tags", [])
#         })
#
#     prompt = f"""
#         Given the following user conversation and a list of incidents, identify the 8 incidents
#         that are most similar or relevant to the conversation.
#
#         User Conversation:
#         "{conversation}"
#
#         Incidents:
#         {json.dumps(incidents_for_llm, indent=4)}
#
#         Please return a JSON array containing only the 'id' of the 8 most similar incidents,
#         ranked from most to least similar.
#         Example: ["incident_id_1", "incident_id_2", "incident_id_3"]
#         """
#     start = time.time()
#     response = client.models.generate_content(
#         model="gemini-2.5-flash-lite-preview-06-17",
#         contents=prompt,
#         config={
#             'response_mime_type': 'application/json',
#             'response_schema': {
#                 "type": "array",
#                 "items": {
#                     "type": "string",
#                     "enum": [incident.get("id") for incident in incidents_data],
#                 },
#             },
#         }
#     )
#
#     # Use the response as a JSON string.
#     print(time.time() - start)
#     print(response.text)
#     return response.text