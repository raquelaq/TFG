import json
import re
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util
import time
import os
import torch
from ..config import *
from app.config import DATA_DIR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(BASE_DIR, "data", "KnowledgeBase.json")

EMBEDDING_CACHE_FILE = str(DATA_DIR / "kb_embeddings.json")
DEFAULT_MODEL_NAME = 'multi-qa-mpnet-base-dot-v1'

model: Optional[SentenceTransformer] = None
KB_CORPUS_EMBEDDINGS: Optional[torch.Tensor] = None
KB_CORPUS_DATA: Optional[List[Dict[str, Any]]] = None

conversation_history_embeddings: Dict[str, List[torch.Tensor]] = {}

_model_initialized = False

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

def preprocess_text(text: str) -> str:
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_embeddings_from_cache(cache_file: str) -> Optional[Dict[str, torch.Tensor]]:
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            return {k: torch.tensor(v) for k, v in loaded_data.items()}
    except Exception as e:
        print(f"Error loading embeddings cache: {e}")
        return None

def save_embeddings_to_cache(embeddings: Dict[str, torch.Tensor], cache_file: str):
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)  # 👈 CLAVE
        serializable_embeddings = {k: v.tolist() for k, v in embeddings.items()}
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_embeddings, f, indent=4)
    except Exception as e:
        print(f"Error saving embeddings cache: {e}")


def get_weighted_context_embedding(user_email: str, new_message_embedding: torch.Tensor, decay_factor: float = 0.8, max_history: int = 5) -> torch.Tensor:
    global conversation_history_embeddings

    if user_email not in conversation_history_embeddings:
        conversation_history_embeddings[user_email] = []

    conversation_history_embeddings[user_email].append(new_message_embedding)

    if len(conversation_history_embeddings[user_email]) > max_history:
        conversation_history_embeddings[user_email] = conversation_history_embeddings[user_email][-max_history:]

    if not conversation_history_embeddings[user_email]:
        return new_message_embedding

    weighted_embeddings = []
    for i, emb in enumerate(reversed(conversation_history_embeddings[user_email])):
        weight = decay_factor**i
        weighted_embeddings.append(emb * weight)

    context_embedding = torch.stack(weighted_embeddings).sum(dim=0)
    return context_embedding / context_embedding.norm()


def initialize_model_and_kb(Route, force_reload=False):

    global model, KB_CORPUS_EMBEDDINGS, KB_CORPUS_DATA, _model_initialized

    if _model_initialized and not force_reload:
        return

    _model_initialized = False
    KB_CORPUS_EMBEDDINGS = None
    KB_CORPUS_DATA = None

    print(f"Attempting to load SentenceTransformer model '{DEFAULT_MODEL_NAME}'...")
    try:
        start = time.time()
        model = SentenceTransformer(DEFAULT_MODEL_NAME, device = 'cpu')
        print(f"SentenceTransformer model loaded successfully in {time.time() - start} seconds.")
    except Exception as e:
        _model_initialized = False
        print(f"CRITICAL ERROR: Failed to load SentenceTransformer model: {e}")
        model = None
        return #

    print("Loading Knowledge Base data and embeddings...")
    data = load_json_data(KB_PATH)
    if not data:
        print("No knowledge base data loaded. KB filtering will be inactive.")
        KB_CORPUS_EMBEDDINGS = None
        KB_CORPUS_DATA = None
        return

    cached_embeddings = load_embeddings_from_cache(Route)
    new_embeddings = {}
    corpus_embeddings_list = []
    corpus_data_ordered = []
    start = time.time()

    for incident in data:
        incident_id = incident['id']
        text_to_embed = preprocess_text(incident.get('description_problem', '') + ' ' + incident.get('title', '') + ' ' + ", ".join(incident.get('keywords_tags', [])))

        if cached_embeddings and incident_id in cached_embeddings:
            corpus_embeddings_list.append(cached_embeddings[incident_id])
        else:
            embedding = model.encode(text_to_embed, convert_to_tensor=True)
            corpus_embeddings_list.append(embedding)
            new_embeddings[incident_id] = embedding
        corpus_data_ordered.append(incident)

    if new_embeddings:
        all_embeddings = cached_embeddings or {}
        all_embeddings.update(new_embeddings)
        save_embeddings_to_cache(all_embeddings, cache_file=Route)

    if corpus_embeddings_list:
        KB_CORPUS_EMBEDDINGS = torch.stack(corpus_embeddings_list)
        KB_CORPUS_DATA = corpus_data_ordered
        print(f"Knowledge Base and embeddings prepared in {time.time() - start} seconds. Total {len(KB_CORPUS_DATA)} entries.")
        _model_initialized = True
    else:
        KB_CORPUS_EMBEDDINGS = None
        KB_CORPUS_DATA = None
        print("No valid embeddings generated for Knowledge Base.")


def get_relevant_incidents_weighted_context(
    user_email: str,
    query: str,
    query_weight: float = 0.7,
    context_weight: float = 0.3,
    decay_factor: float = 0.8,
    max_history: int = 5,
    top_n: int = 1
) -> List[Dict[str, Any]]:
    if not isinstance(KB_CORPUS_DATA, list):
        return []
    if model is None:
        print("Error: SentenceTransformer model is not loaded. Cannot perform filtering.")
        return []
    if KB_CORPUS_EMBEDDINGS is None or KB_CORPUS_DATA is None:
        print("Error: Knowledge Base embeddings not loaded. Cannot perform filtering.")
        return []

    incidents = []
    if os.path.exists(KB_PATH + 'conversation_store.json'):
        with open(KB_PATH + 'conversation_store.json', 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            incidents = data.get(user_email, {}).get("Incidents", [])
    else:
        incidents = []

    query_embedding = model.encode(query, convert_to_tensor=True)

    context_embedding = get_weighted_context_embedding(user_email, query_embedding, decay_factor, max_history)

    query_scores = util.dot_score(query_embedding.unsqueeze(0), KB_CORPUS_EMBEDDINGS)[0]
    context_scores = util.dot_score(context_embedding.unsqueeze(0), KB_CORPUS_EMBEDDINGS)[0]

    combined_scores = (query_scores * query_weight) + (context_scores * context_weight)

    incident_scores = list(zip(KB_CORPUS_DATA, combined_scores.tolist()))
    sorted_incidents = sorted(incident_scores, key=lambda x: x[1], reverse=True)

    past_incidents_from_kb = []

    for item in KB_CORPUS_DATA:
        if "id" in item and item["id"] in incidents:
            past_incidents_from_kb.append(item)
    unique_incident_ids = set()
    top_results = []

    for incident_dict, score in sorted_incidents[:top_n]:
        if "id" in incident_dict and incident_dict["id"] not in unique_incident_ids:
            top_results.append(incident_dict)
            unique_incident_ids.add(incident_dict["id"])

    for incident_dict in past_incidents_from_kb:
        if "id" in incident_dict and incident_dict["id"] not in unique_incident_ids:
            top_results.append(incident_dict)
            unique_incident_ids.add(incident_dict["id"])

    return top_results

def rebuild_embeddings(cache_file: str = EMBEDDING_CACHE_FILE) -> None:
    global model, KB_CORPUS_EMBEDDINGS, KB_CORPUS_DATA

    print("🔁 Rebuilding Knowledge Base embeddings...")

    if model is None:
        try:
            print(f"Loading model '{DEFAULT_MODEL_NAME}' inside rebuild_embeddings...")
            model = SentenceTransformer(DEFAULT_MODEL_NAME)
        except Exception as e:
            print(f"CRITICAL ERROR: cannot load model in rebuild_embeddings: {e}")
            KB_CORPUS_EMBEDDINGS = None
            KB_CORPUS_DATA = None
            return

    data = load_json_data(KB_PATH)
    if not data:
        print("No KB data found in rebuild_embeddings. Globals set to None.")
        KB_CORPUS_EMBEDDINGS = None
        KB_CORPUS_DATA = None
        return

    embeddings_dict: Dict[Any, torch.Tensor] = {}
    corpus_embeddings_list: List[torch.Tensor] = []
    corpus_data_ordered: List[Dict[str, Any]] = []

    start = time.time()

    for incident in data:
        incident_id = incident["id"]
        text_to_embed = preprocess_text(
            incident.get("description_problem", "")
            + " "
            + incident.get("title", "")
            + " "
            + ", ".join(incident.get("keywords_tags", []))
        )

        emb = model.encode(text_to_embed, convert_to_tensor=True)
        embeddings_dict[incident_id] = emb
        corpus_embeddings_list.append(emb)
        corpus_data_ordered.append(incident)

    save_embeddings_to_cache(embeddings_dict, cache_file=cache_file)

    KB_CORPUS_EMBEDDINGS = torch.stack(corpus_embeddings_list)
    KB_CORPUS_DATA = corpus_data_ordered

    print(
        f"✅ Rebuild completado: {len(KB_CORPUS_DATA)} entradas, "
        f"{time.time() - start:.2f} segundos."
    )