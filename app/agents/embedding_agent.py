import os
import numpy as np
from app.services.embeddings_index import EmbeddingsIndex

SIMILARITY_THRESHOLD = float(os.getenv("EMB_SIM_THRESHOLD", "0.75"))
TOP_K = int(os.getenv("EMB_TOP_K", "3"))

_index: EmbeddingsIndex | None = None

def get_index() -> EmbeddingsIndex:
    global _index
    if _index is None:
        _index = EmbeddingsIndex()
        _index.initialize(force_rebuild=False)
    return _index

def _format_incident_steps(incident: dict) -> str:
    pasos = incident.get("resolution_guide_llm", {}).get("diagnostic_steps", [])
    if not pasos:
        return "No he encontrado pasos detallados en la guía para este caso."
    out = []
    for step in pasos:
        t = step.get("title", "").strip()
        a = step.get("user_action", "").strip()
        if t:
            out.append(f"**{t}**\n{a}")
        else:
            out.append(a)
    return "\n\n".join(out)

def responder_con_embeddings_custom(pregunta_usuario: str) -> dict:
    idx = get_index()
    resultados = idx.search(pregunta_usuario, top_k=TOP_K)

    for r in resultados:
        sim = r.get("_similarity", 0.0)
        if sim >= SIMILARITY_THRESHOLD:
            respuesta = (
                f"📘 He encontrado una guía técnica útil (similitud {sim:.2f}):\n\n"
                f"{_format_incident_steps(r)}"
            )
            return {
                "pregunta_usuario": pregunta_usuario,
                "similitud": sim,
                "solved": True,
                "respuesta": respuesta
            }

    best = resultados[0] if resultados else {}
    sim = best.get("_similarity", 0.0)
    return {
        "pregunta_usuario": pregunta_usuario,
        "similitud": sim,
        "solved": False,
        "respuesta": "No se encontró una solución clara. ¿Quieres que creemos un ticket?"
    }
