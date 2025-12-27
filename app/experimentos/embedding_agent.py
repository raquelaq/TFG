import os
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

KB_JSON_PATH = "./app/data/KnowledgeBase.json"
MODEL_NAME = "C:/Users/ralme/OneDrive/Escritorio/4_GCID/TFG/corporate-api-genia-soportia/app/embedding_model_custom"
SIMILARITY_THRESHOLD = 0.75
TOP_K = 3

print("Cargando modelo de embeddings")
embedding_model = SentenceTransformer(MODEL_NAME)
print("✅ Modelo cargado correctamente.")

def normalizar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto.strip()

def cargar_conocimiento_desde_json():
    with open(KB_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data if isinstance(data, list) else [data]

def preparar_corpus_y_embeddings():
    kb_data = cargar_conocimiento_desde_json()
    corpus = []
    ids = []

    for entry in kb_data:
        symptoms = " ".join(entry.get("symptoms", []))
        tags = " ".join(entry.get("keywords_tags", []))
        diagnostics = " ".join([
            step.get("title", "") + " " + step.get("user_action", "")
            for step in entry.get("resolution_guide_llm", {}).get("diagnostic_steps", [])
        ])
        escalation = entry.get("escalation_criteria", "")

        base = f"{entry.get('title', '')}. {entry.get('description_problem', '')} {symptoms} {tags} {diagnostics} {escalation}"

        corpus.append(base)
        ids.append(entry.get("id"))

    embeddings = embedding_model.encode(corpus, convert_to_tensor=False)
    return corpus, embeddings, kb_data, ids

CORPUS, CORPUS_EMBEDDINGS, KB_RAW_DATA, KB_IDS = preparar_corpus_y_embeddings()

def responder_con_embeddings_custom(pregunta_usuario: str) -> dict:
    pregunta_normalizada = normalizar_texto(pregunta_usuario)
    embedding_pregunta = embedding_model.encode([pregunta_normalizada])[0]
    similitudes = cosine_similarity([embedding_pregunta], CORPUS_EMBEDDINGS)[0]

    top_indices = np.argsort(similitudes)[::-1][:TOP_K]

    for idx in top_indices:
        similitud = float(similitudes[idx])
        if similitud >= SIMILARITY_THRESHOLD:
            incidente = KB_RAW_DATA[idx]
            pasos = incidente.get("resolution_guide_llm", {}).get("diagnostic_steps", [])
            respuesta = f"📘 He encontrado una guía técnica útil (similitud {similitud:.2f}):\n\n"

            for paso in pasos:
                respuesta += f"**{paso['title']}**\n{paso['user_action']}\n\n"

            print(f"✅ Resuelto con embeddings (match índice {idx} - similitud {similitud:.4f})")
            return {
                "pregunta_usuario": pregunta_usuario,
                "similitud": similitud,
                "solved": True,
                "respuesta": respuesta.strip()
            }

    print("✅ Resuelto con embeddings (sin coincidencias claras)")
    return {
        "pregunta_usuario": pregunta_usuario,
        "similitud": max(similitudes),
        "solved": False,
        "respuesta": "No se encontró una solución clara. ¿Quieres que creemos un ticket?"
    }