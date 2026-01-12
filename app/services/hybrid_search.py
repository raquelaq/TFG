import os
import json
import re
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from rapidfuzz import process

"""
Módulo de búsqueda híbrida (BM25 + embeddings).
Usado únicamente cuando el usuario selecciona 'Modelo ML (embeddings)'.
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "..", "data", "KnowledgeBase.json")

kb = None
kb_filtrada = None
KB_VOCAB = set()
bm25 = None
embeddings_kb = None
model = None
texts = None

MODEL_NAME = "intfloat/e5-base-v2"

INFORMAL_MAP = {
    "no va": "no funciona",
    "no tira": "no funciona",
    "no conecta": "problema de conexión",
    "da error": "error",
    "no abre": "no abre aplicación",
    "se queda pillado": "aplicación bloqueada"
}

TECH_CORE_TERMS = {
    "impresora", "vpn", "correo", "email", "red", "servidor",
    "usuario", "acceso", "error", "configuración", "conexión",
    "aplicación", "sistema", "ordenador", "pc"
}


def initialize_hybrid_search():
    global kb, kb_filtrada, KB_VOCAB, bm25, embeddings_kb, model, texts

    if kb is not None:
        return

    if not os.path.exists(KB_PATH):
        raise FileNotFoundError(f"No se encontró la KB en {KB_PATH}")

    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
            if not isinstance(kb, list):
                kb = []
    except json.JSONDecodeError:
        print("KB corrupta en hybrid_search")
        kb = []

    kb_filtrada = [
        item for item in kb
        if not item["title"].lower().startswith("solicitud")
    ]

    for item in kb_filtrada:
        KB_VOCAB.update(re.findall(r'\w+', item["title"].lower()))
        KB_VOCAB.update(item["keywords_tags"])

    texts = [
        f"{item['description_problem']} "
        f"{' '.join(item['symptoms'])} "
        f"{' '.join(item['keywords_tags'])}"
        for item in kb_filtrada
    ]

    tokenized_texts = [re.findall(r'\w+', text.lower()) for text in texts]
    bm25 = BM25Okapi(tokenized_texts)

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.eval()
    with torch.no_grad():
        embeddings_kb = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

def soft_spellcheck(query: str) -> str:
    tokens = query.split()
    fixed = []
    for t in tokens:
        if len(t) < 5:
            fixed.append(t)
        else:
            match, score, _ = process.extractOne(t, KB_VOCAB)
            fixed.append(match if score > 85 else t)
    return " ".join(fixed)


def normalize_query(query: str) -> str:
    q = query.lower()
    q = re.sub(r"[^\w\sáéíóúñü]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def expand_informal_language(query: str) -> str:
    expanded = []
    for k, v in INFORMAL_MAP.items():
        if k in query:
            expanded.append(v)
    return query + " " + " ".join(expanded)


def is_out_of_domain(query: str) -> bool:
    return not any(term in query for term in TECH_CORE_TERMS)


def buscar_hibrido(query: str, alpha: float = 0.25, top_k: int = 3):
    initialize_hybrid_search()

    query_norm = normalize_query(query)

    if is_out_of_domain(query_norm):
        return []

    query_augmented = expand_informal_language(query_norm)
    query_augmented = soft_spellcheck(query_augmented)

    tokens = re.findall(r'\w+', query_augmented.lower())
    bm25_scores = np.array(bm25.get_scores(tokens))

    bm25_norm = bm25_scores / bm25_scores.max() if bm25_scores.max() > 0 else bm25_scores

    query_emb = model.encode([query_augmented], normalize_embeddings=True)[0]
    sim_scores = embeddings_kb @ query_emb
    sim_norm = (sim_scores + 1) / 2

    hybrid_score = alpha * bm25_norm + (1 - alpha) * sim_norm

    top_idx = np.argsort(hybrid_score)[::-1][:top_k]

    resultados = []
    for i in top_idx:
        item = kb_filtrada[i]
        resultados.append({
            "id": item["id"],
            "title": item["title"],
            "score_hybrid": float(hybrid_score[i]),
            "score_cosine": float(sim_scores[i]),
            "score_bm25": float(bm25_scores[i])
        })

    return resultados

def get_kb_item_by_id(incidente_id: str):
    initialize_hybrid_search()
    for item in kb_filtrada:
        if item.get("id") == incidente_id:
            return item
    return None